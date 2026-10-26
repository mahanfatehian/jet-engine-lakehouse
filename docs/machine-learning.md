# Machine learning deep dive

The physics that generates the data, the labels and features built from it, and how the RUL model
is trained, validated, promoted and served. For the quick tour, start with the
[README](../README.md).

## 1. Engine physics: a learnable ground truth

Real engine run-to-failure data is scarce, so the simulator
([`simulator/engine_generator.py`](../simulator/engine_generator.py)) generates it from a hidden,
physics-inspired **wear factor** `w(t)` that runs from `0` (fresh overhaul) to `1` (failure). The
model never sees `w`: it has to infer degradation from noisy sensors, exactly like a real engine
health monitoring system.

**One lifecycle** (overhaul → failure) draws its own parameters from a stable hash of
`(SIM_SEED, engine_id, lifecycle_id)`:

```text
expected life   L ~ Normal(μ, 0.18 μ), clipped to [0.55 μ, 1.6 μ]        μ = SIM_MEAN_LIFE_CYCLES (180)
wear curve      c(t) = (t / L)^k                         k ~ U(1.8, 2.8)   polynomial  (p = 0.5)
                c(t) = (e^(b t / L) − 1) / (e^b − 1)     b ~ U(2.5, 4.0)   exponential (p = 0.5)
per pulse       w_i = min(1, Σ [c(t_i) − c(t_i−1)] · s_i),   s_i ~ LogNormal with mean 1, σ = 0.25
```

Both curve families are convex: damage accrues slowly in a healthy engine and accelerates towards
end of life (crack growth, coating loss), which is exactly the shape that makes RUL learnable.

**Sensor response** (B = per-engine baseline with ±1.5 % manufacturing offset, θ = shared operating
condition, an AR(1) process around 1.0, z = Gaussian noise, m = failure-mode sensitivity):

| Sensor | Model | Physical story |
|---|---|---|
| EGT | `B(1 + 0.8(θ−1)) + 110·m·w^2.2 + (4 + 6w)·z` | hot-section efficiency loss, non-linear drift |
| Core temperature | `B(1 + 0.6(θ−1)) + 70·m·w^1.8 + (3 + 4w)·z` | same mechanism, smaller amplitude |
| Vibration | `B(1 + 3.5·m·w^2.5)·e^((0.05 + 0.2w)·z)` + rare spikes (p = 0.002 + 0.03w³) | bearing wear: amplitude **and** variance grow |
| Oil pressure | `B(1 − 0.02w − 0.12·m·max(0, (w−0.7)/0.3)²) + (0.4 + w)·z` − dips (p = 0.04w⁴) | subtle drop shortly before failure |
| Fuel flow | `B·θ^1.5·(1 + 0.05·m·w^1.6) + 0.006·B·z` | specific fuel consumption rises |
| Core speed | `B·θ·(1 + 0.012·m·w^1.5) + 0.002·B·z` | the core spins faster to hold thrust |

Each lifecycle draws a **failure mode** that amplifies its dominant channels (×1.5, others ×0.8):
`HPT_BLADE_EROSION` (EGT, core temperature), `BEARING_DEGRADATION` (vibration), `OIL_SYSTEM_LEAK`
(oil pressure), `COMPRESSOR_FOULING` (fuel flow, rpm, EGT). Every channel still drifts, so RUL stays
learnable whatever the mode.

```mermaid
stateDiagram-v2
    [*] --> HEALTHY
    HEALTHY --> WARNING: w ≥ 0.50
    WARNING --> CRITICAL: w ≥ 0.80
    CRITICAL --> FAILURE: w = 1.00, work order opened
    FAILURE --> MAINTENANCE_RESET: repair done, w back to 0
    MAINTENANCE_RESET --> HEALTHY: first pulse of the new lifecycle
```

With the defaults a lifecycle lasts ~180 cycles (CV 0.17, observed range 98–285) of 10 pulses
each, about 30 minutes at 1 Hz. Measured on the live stack, average EGT rises from 657 °C (healthy)
to 786 °C at failure, vibration from 1.33 to 5.98 mm/s, and oil pressure falls from 52.4 to 45.2 psi.

## 2. Labels

For every pulse of a **completed** lifecycle (one that ended in a FAILURE pulse):

| Label | Definition |
|---|---|
| `rul_cycles` | `failure_cycle − cycle_number`, counting down to exactly 0 at the failure pulse |
| `rul_cycles_capped` | `least(rul_cycles, RUL_CAP_CYCLES)` with cap 125: early-life RUL is not identifiable from a healthy engine's sensors, so the regression target is piecewise linear (standard C-MAPSS practice) |
| `failure_within_horizon` | `rul_cycles ≤ RUL_FAILURE_HORIZON_CYCLES` (25): the "remove at the next stop" signal |

In-progress lifecycles are right-censored and never used for training, only for inference. dbt and
the trainer read the cap and the horizon from the same environment variables, and the trainer
warns if the labels it recomputes disagree with the mart's.

## 3. Features

46 numeric features per pulse, computed in dbt
([`int_sensor_rolling_features`](../analytics/models/silver/int_sensor_rolling_features.sql)) and
shared verbatim by training and inference, so there is no train/serve skew:

| Family | Columns |
|---|---|
| Age | `cycle_number` (cycles since the last overhaul) |
| Raw sensors (6) | `rpm`, `core_temp_c`, `egt_c`, `vibration_mm_s`, `fuel_flow_kg_h`, `oil_pressure_psi` |
| Rolling windows (24) | mean, std, min and max over **5 and 15 minutes** for vibration, EGT and oil pressure |
| Dynamics (9) | `*_delta_1` (pulse-to-pulse change), `*_roc_5m` / `*_roc_15m` (change per minute across the window) |
| Drift (6) | EGT and core-temperature drift, vibration ratio, oil-pressure drop, fuel-flow and rpm drift in %, all against an *expanding* baseline of the lifecycle's first 10 cycles |

Plus `engine_model` one-hot (categories persisted with the model). Windows are time ranges
(`RANGE BETWEEN 300 PRECEDING AND CURRENT ROW` over epoch seconds) partitioned by engine
**and lifecycle**, so a repair never blends two lifecycles. A guard in
[`feature_pipeline.py`](../data_science/feature_pipeline.py) refuses ground truth (`wear_factor`),
status columns, identifiers and labels as features.

## 4. Training and validation

[`train_rul_model.py`](../data_science/train_rul_model.py):

1. **Load** `mart_rul_training_features` from ClickHouse (default) or the newest gold snapshot in
   MinIO (`--source minio`). Exit code 3 (skipped) below `MIN_TRAINING_LIFECYCLES` completed lifecycles.
2. **Split by lifecycle.** A lifecycle never appears on both sides. The assignment is a deterministic
   SHA-256 of the lifecycle key (~25 % test, ~20 % of the rest for validation), so a lifecycle
   never migrates from train to test as data grows, and the champion is never scored on its own
   training data.
3. **Regressor:** XGBoost (`max_depth` 5, `learning_rate` 0.03, subsampling 0.8, L2 2.0, up to 2 000
   trees with early stopping on the validation lifecycles) on `rul_cycles_capped`.
4. **Classifier:** scikit-learn `HistGradientBoostingClassifier` for `failure_within_horizon`, with
   its decision threshold tuned for F1 on the validation lifecycles. The threshold, not class
   re-weighting, handles the imbalance, so `failure_probability` is not inflated by sample weights.
5. **Evaluate on the holdout lifecycles:**

| Metric | Meaning |
|---|---|
| RMSE, MAE, R² | against the capped target |
| late-life RMSE | rows with true RUL ≤ cap: the part of life where the prediction matters |
| NASA score | asymmetric C-MAPSS score, mean per row: late predictions (d > 0) cost `e^(d/10) − 1`, early ones `e^(−d/13) − 1` |
| F1, precision, recall, ROC-AUC | imminent-failure classifier at the tuned threshold |

Example run on the 24 backfilled lifecycles (21 train/validation, 3 test): **RMSE 9.6 cycles,
MAE 6.8, R² 0.947, late-life RMSE 7.2, ROC-AUC 0.998**.

## 5. Promotion, artifacts and the registry

- **Champion/challenger.** When a production model exists it is re-scored on the challenger's
  holdout; with `MODEL_PROMOTION_POLICY=better` the challenger is promoted only if its RMSE is lower
  (`always` promotes unconditionally). `production.json` is replaced with a single atomic write.
- **Artifacts** go to `s3://lakehouse/ml-artifacts/rul_model/rul-<UTC timestamp>/`:
  `model_bundle.joblib` (both models, threshold, feature list, categories, horizon, cap, library
  versions), `regressor.json` (portable XGBoost format), `metadata.json`, `metrics.json` and
  `feature_importance.csv`.
- **Registry.** Every run, promoted or not, appends a row to `ml_model_runs` with data volume,
  holdout metrics, the promotion decision and the artifact URI.
- **Scheduled retraining** uses `--skip-if-unchanged`: no new completed lifecycle, no new model.

## 6. Inference

[`predict_rul.py`](../data_science/predict_rul.py) loads the model named by `production.json`,
takes the newest feature row of every engine (`int_sensor_rolling_features FINAL`, `LIMIT 1 BY
engine_id`) and writes to `gold_engine_rul_predictions`:

| Column | Meaning |
|---|---|
| `predicted_rul_cycles` | regressor output clipped to `[0, cap]` |
| `predicted_rul_hours` | cycles × `FLIGHT_HOURS_PER_CYCLE` (2.0) |
| `failure_probability`, `failure_within_horizon` | classifier probability and the tuned decision |
| `risk_band` | `HIGH` if a failure is predicted within the horizon or RUL ≤ horizon; `MEDIUM` if the probability exceeds half the threshold or RUL ≤ 2 × horizon; otherwise `LOW` |
| sensor columns, `operational_status` | the readings the prediction was made from |

`--mode live --interval-seconds N` keeps scoring in a loop; the orchestrator instead calls
`--mode batch` after every dbt cycle.

## 7. Known limitations

- The data is simulated: the physics is plausible, not calibrated against a real engine type.
- The mart holds one row per pulse of every completed lifecycle (~0.5 M rows per simulated day);
  for long-running stacks add retention or train on a window of recent lifecycles.
- `model_bundle.joblib` is a pickle: load it only from a bucket that only the training job can write.
