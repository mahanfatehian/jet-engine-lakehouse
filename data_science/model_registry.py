"""Model artifacts in the lakehouse bucket, the production pointer, the ml_model_runs
registry and the champion/challenger promotion rule.

Layout below ML_ARTIFACT_PREFIX (interface contract, section 5)::

    <prefix>/<model_version>/model_bundle.joblib    both models + feature/label contract
    <prefix>/<model_version>/regressor.json         XGBoost regressor, native JSON format
    <prefix>/<model_version>/metadata.json          data, split, parameters, versions, promotion
    <prefix>/<model_version>/metrics.json           holdout + validation metrics, champion comparison
    <prefix>/<model_version>/feature_importance.csv
    <prefix>/production.json                        pointer to the model that serves predictions

Version artifacts are written before the pointer moves, so production.json only
ever references a complete artifact set.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, Protocol

import joblib
import pandas as pd
from artifact_store import JSON_CONTENT_TYPE, ArtifactNotFoundError, ArtifactStore
from ds_settings import PROMOTION_POLICIES
from lakehouse_io import ClickHouseGateway
from rul_modeling import RulModelBundle

BUNDLE_FILE = "model_bundle.joblib"
REGRESSOR_FILE = "regressor.json"
METADATA_FILE = "metadata.json"
METRICS_FILE = "metrics.json"
IMPORTANCE_FILE = "feature_importance.csv"
PRODUCTION_POINTER_FILE = "production.json"
MODEL_VERSION_FORMAT = "rul-%Y%m%dT%H%M%SZ"
ML_MODEL_RUNS_TABLE = "ml_model_runs"

logger = logging.getLogger(__name__)


def new_model_version(trained_at: datetime) -> str:
    """Model version for a training run started at ``trained_at`` (UTC)."""
    if trained_at.tzinfo is None:
        raise ValueError("trained_at must be timezone-aware")
    return trained_at.astimezone(UTC).strftime(MODEL_VERSION_FORMAT)


@dataclass(frozen=True)
class ArtifactLayout:
    """Key layout of the model artifacts below ``prefix`` (ML_ARTIFACT_PREFIX)."""

    prefix: str

    def version_prefix(self, model_version: str) -> str:
        """Key prefix (no trailing slash) holding one model version's artifacts."""
        return f"{self.prefix}/{model_version}"

    def artifact_key(self, model_version: str, file_name: str) -> str:
        """Key of one artifact file of ``model_version``."""
        return f"{self.version_prefix(model_version)}/{file_name}"

    @property
    def production_key(self) -> str:
        """Key of the production pointer."""
        return f"{self.prefix}/{PRODUCTION_POINTER_FILE}"


@dataclass(frozen=True)
class ProductionPointer:
    """Content of production.json."""

    model_version: str
    artifact_prefix: str
    promoted_at: str
    metrics: dict[str, Any]
    feature_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        """JSON document of the pointer."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, payload: Any) -> ProductionPointer:
        """Parse production.json, rejecting documents without the required keys."""
        if not isinstance(payload, dict) or not {"model_version", "artifact_prefix"} <= payload.keys():
            raise ValueError("production.json must contain model_version and artifact_prefix")
        return cls(
            model_version=str(payload["model_version"]),
            artifact_prefix=str(payload["artifact_prefix"]).strip("/"),
            promoted_at=str(payload.get("promoted_at", "")),
            metrics=dict(payload.get("metrics") or {}),
            feature_columns=list(payload.get("feature_columns") or []),
        )


@dataclass(frozen=True)
class ModelArtifacts:
    """Everything published for one model version."""

    bundle: RulModelBundle
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    feature_importance: pd.DataFrame


def serialize_bundle(bundle: RulModelBundle) -> bytes:
    """Serialise a bundle to joblib bytes."""
    buffer = io.BytesIO()
    joblib.dump(bundle.to_payload(), buffer, compress=3)
    return buffer.getvalue()


def deserialize_bundle(data: bytes) -> RulModelBundle:
    """Load a bundle from joblib bytes (artifacts are trusted: written by this platform only)."""
    return RulModelBundle.from_payload(joblib.load(io.BytesIO(data)))


def publish_artifacts(store: ArtifactStore, layout: ArtifactLayout, artifacts: ModelArtifacts) -> str:
    """Upload every artifact of one model version; return the URI of its prefix."""
    version = artifacts.bundle.model_version
    regressor_json = bytes(artifacts.bundle.regressor.get_booster().save_raw(raw_format="json"))
    store.put_bytes(layout.artifact_key(version, REGRESSOR_FILE), regressor_json, JSON_CONTENT_TYPE)
    store.put_json(layout.artifact_key(version, METADATA_FILE), artifacts.metadata)
    store.put_json(layout.artifact_key(version, METRICS_FILE), artifacts.metrics)
    store.put_bytes(
        layout.artifact_key(version, IMPORTANCE_FILE),
        artifacts.feature_importance.to_csv(index=False).encode("utf-8"),
        "text/csv",
    )
    store.put_bytes(layout.artifact_key(version, BUNDLE_FILE), serialize_bundle(artifacts.bundle))
    return store.uri(layout.version_prefix(version)) + "/"


def read_production_pointer(store: ArtifactStore, layout: ArtifactLayout) -> ProductionPointer | None:
    """Return the current production pointer, or None when no model was promoted yet."""
    try:
        payload = store.get_json(layout.production_key)
    except ArtifactNotFoundError:
        return None
    return ProductionPointer.from_dict(payload)


def write_production_pointer(
    store: ArtifactStore, layout: ArtifactLayout, pointer: ProductionPointer
) -> None:
    """Replace production.json in one atomic write."""
    store.put_json(layout.production_key, pointer.to_dict())


def load_production_bundle(store: ArtifactStore, pointer: ProductionPointer) -> RulModelBundle:
    """Download and validate the bundle a production pointer references."""
    bundle = deserialize_bundle(store.get_bytes(f"{pointer.artifact_prefix}/{BUNDLE_FILE}"))
    if bundle.model_version != pointer.model_version:
        raise ValueError(
            f"production.json references {pointer.model_version} but its bundle is {bundle.model_version}"
        )
    return bundle


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the champion/challenger comparison."""

    promote: bool
    reason: str


def decide_promotion(policy: str, challenger_rmse: float, champion_rmse: float | None) -> PromotionDecision:
    """Apply MODEL_PROMOTION_POLICY to holdout RMSEs measured on the same lifecycles.

    ``champion_rmse`` is None when there is no usable production model, in which
    case the challenger is always promoted.
    """
    if policy not in PROMOTION_POLICIES:
        raise ValueError(f"unknown promotion policy {policy!r}")
    if champion_rmse is None:
        return PromotionDecision(True, "no usable production model to compare against")
    comparison = f"challenger RMSE {challenger_rmse:.3f} vs champion RMSE {champion_rmse:.3f}"
    if policy == "always":
        return PromotionDecision(True, f"policy 'always' ({comparison})")
    if challenger_rmse < champion_rmse:
        return PromotionDecision(True, f"challenger is better ({comparison})")
    return PromotionDecision(False, f"challenger is not better ({comparison})")


ML_MODEL_RUNS_COLUMNS = (
    "model_version",
    "trained_at",
    "data_source",
    "train_rows",
    "test_rows",
    "train_lifecycles",
    "test_lifecycles",
    "rmse",
    "mae",
    "r2",
    "nasa_score",
    "clf_f1",
    "clf_precision",
    "clf_recall",
    "clf_roc_auc",
    "promoted",
    "artifact_uri",
    "params_json",
)


@dataclass(frozen=True)
class ModelRunRecord:
    """One row of ml_model_runs (``train_*`` counts include the validation lifecycles)."""

    model_version: str
    trained_at: datetime
    data_source: str
    train_rows: int
    test_rows: int
    train_lifecycles: int
    test_lifecycles: int
    rmse: float
    mae: float
    r2: float
    nasa_score: float
    clf_f1: float
    clf_precision: float
    clf_recall: float
    clf_roc_auc: float
    promoted: bool
    artifact_uri: str
    params_json: str

    def as_row(self) -> list[Any]:
        """Values in ML_MODEL_RUNS_COLUMNS order, typed for ClickHouse."""
        return [
            int(self.promoted) if name == "promoted" else getattr(self, name)
            for name in ML_MODEL_RUNS_COLUMNS
        ]


@dataclass(frozen=True)
class RunSummary:
    """The facts of a past training run that --skip-if-unchanged needs."""

    model_version: str
    lifecycles: int


class RunRegistry(Protocol):
    """Store of training runs (ClickHouse ml_model_runs in production)."""

    def latest_run(self) -> RunSummary | None:
        """Most recent recorded run, if any."""
        ...

    def record_run(self, record: ModelRunRecord) -> None:
        """Append one run."""
        ...


class ClickHouseRunRegistry:
    """RunRegistry backed by the ml_model_runs table."""

    def __init__(self, gateway: ClickHouseGateway) -> None:
        self._gateway = gateway

    def latest_run(self) -> RunSummary | None:
        """Most recent row of ml_model_runs."""
        rows = self._gateway.query_rows(
            f"SELECT model_version, train_lifecycles + test_lifecycles FROM {ML_MODEL_RUNS_TABLE} "
            "ORDER BY trained_at DESC, model_version DESC LIMIT 1"
        )
        return RunSummary(model_version=str(rows[0][0]), lifecycles=int(rows[0][1])) if rows else None

    def record_run(self, record: ModelRunRecord) -> None:
        """Insert one row into ml_model_runs."""
        self._gateway.insert_rows(ML_MODEL_RUNS_TABLE, [record.as_row()], ML_MODEL_RUNS_COLUMNS)
        logger.info("recorded run %s in %s", record.model_version, ML_MODEL_RUNS_TABLE)
