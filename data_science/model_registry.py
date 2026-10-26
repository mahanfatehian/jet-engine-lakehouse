"""Model artifacts in the lakehouse bucket, the production pointer and the ml_model_runs registry.

Layout below ML_ARTIFACT_PREFIX (interface contract, section 5)::

    <prefix>/<model_version>/model_bundle.joblib    both models + feature/label/calibration contract
    <prefix>/<model_version>/regressor.json         XGBoost regressor (served trees only), native JSON
    <prefix>/<model_version>/metadata.json          data, split, parameters, versions, promotion, digest
    <prefix>/<model_version>/metrics.json           holdout + validation metrics, champion comparison
    <prefix>/<model_version>/feature_importance.csv
    <prefix>/production.json                        pointer to the model that serves predictions

Version artifacts are written before the pointer moves, so production.json only
ever references a complete artifact set.

Integrity: production.json carries ``bundle_sha256``, the SHA-256 of model_bundle.joblib
recorded at publish time (also in metadata.json and ml_model_runs.params_json). The bundle
is a pickle, so its digest is verified BEFORE it is unpickled, and a pointer without a
digest or a bundle that does not match is refused. This catches a bundle that was replaced,
truncated or corrupted after publication. Remaining trust assumption: the pointer itself
is trusted. Whoever can write production.json can also write a matching digest, so the
bucket must stay writable by the training job only (dedicated credentials, no anonymous
or shared root keys); defending against such a writer needs a signature whose key lives
outside the bucket, or a non-pickle model format.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import logging
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, Protocol

import joblib
import pandas as pd
from artifact_store import JSON_CONTENT_TYPE, ArtifactNotFoundError, ArtifactStore
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


class ModelIntegrityError(RuntimeError):
    """A model bundle cannot be verified against the digest production.json recorded for it."""


def new_model_version(trained_at: datetime) -> str:
    """Model version for a training run started at ``trained_at`` (UTC)."""
    if trained_at.tzinfo is None:
        raise ValueError("trained_at must be timezone-aware")
    return trained_at.astimezone(UTC).strftime(MODEL_VERSION_FORMAT)


def bundle_digest(data: bytes) -> str:
    """Hex SHA-256 of serialized bundle bytes."""
    return hashlib.sha256(data).hexdigest()


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
    """Content of production.json (``bundle_sha256`` is None for pointers written before digests)."""

    model_version: str
    artifact_prefix: str
    promoted_at: str
    metrics: dict[str, Any]
    feature_columns: list[str]
    bundle_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON document of the pointer."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, payload: Any) -> ProductionPointer:
        """Parse production.json, rejecting documents without the required keys."""
        if not isinstance(payload, dict) or not {"model_version", "artifact_prefix"} <= payload.keys():
            raise ValueError("production.json must contain model_version and artifact_prefix")
        digest = str(payload.get("bundle_sha256") or "").strip().lower()
        return cls(
            model_version=str(payload["model_version"]),
            artifact_prefix=str(payload["artifact_prefix"]).strip("/"),
            promoted_at=str(payload.get("promoted_at", "")),
            metrics=dict(payload.get("metrics") or {}),
            feature_columns=list(payload.get("feature_columns") or []),
            bundle_sha256=digest or None,
        )


@dataclass(frozen=True)
class ModelArtifacts:
    """Everything published for one model version."""

    bundle: RulModelBundle
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    feature_importance: pd.DataFrame


@dataclass(frozen=True)
class PublishedModel:
    """Where a model version was published and the digest of its bundle."""

    uri: str
    bundle_sha256: str


def serialize_bundle(bundle: RulModelBundle) -> bytes:
    """Serialise a bundle to joblib bytes."""
    buffer = io.BytesIO()
    joblib.dump(bundle.to_payload(), buffer, compress=3)
    return buffer.getvalue()


def deserialize_bundle(data: bytes) -> RulModelBundle:
    """Unpickle a bundle. Only call this on bytes whose digest was verified (see load_production_bundle)."""
    return RulModelBundle.from_payload(joblib.load(io.BytesIO(data)))


def publish_artifacts(
    store: ArtifactStore, layout: ArtifactLayout, artifacts: ModelArtifacts
) -> PublishedModel:
    """Upload every artifact of one model version; metadata.json records the bundle digest."""
    version = artifacts.bundle.model_version
    bundle_bytes = serialize_bundle(artifacts.bundle)
    digest = bundle_digest(bundle_bytes)
    # The regressor holds only the trees up to its best iteration, so the native JSON
    # reproduces the served predictions with a plain xgboost.Booster.
    regressor_json = bytes(artifacts.bundle.regressor.get_booster().save_raw(raw_format="json"))
    store.put_bytes(layout.artifact_key(version, REGRESSOR_FILE), regressor_json, JSON_CONTENT_TYPE)
    store.put_json(
        layout.artifact_key(version, METADATA_FILE), {**artifacts.metadata, "bundle_sha256": digest}
    )
    store.put_json(layout.artifact_key(version, METRICS_FILE), artifacts.metrics)
    store.put_bytes(
        layout.artifact_key(version, IMPORTANCE_FILE),
        artifacts.feature_importance.to_csv(index=False).encode("utf-8"),
        "text/csv",
    )
    store.put_bytes(layout.artifact_key(version, BUNDLE_FILE), bundle_bytes)
    return PublishedModel(uri=store.uri(layout.version_prefix(version)) + "/", bundle_sha256=digest)


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
    """Download the bundle a production pointer references, verify its digest, then unpickle it.

    Raises ModelIntegrityError, before anything is unpickled, when the pointer has no
    ``bundle_sha256`` or the downloaded bytes do not match it.
    """
    key = f"{pointer.artifact_prefix}/{BUNDLE_FILE}"
    if not pointer.bundle_sha256:
        raise ModelIntegrityError(
            f"production.json ({pointer.model_version}) has no bundle_sha256: refusing to unpickle "
            f"an unverified {store.uri(key)}; retrain to publish a verifiable model"
        )
    data = store.get_bytes(key)
    actual = bundle_digest(data)
    if not hmac.compare_digest(actual, pointer.bundle_sha256):
        raise ModelIntegrityError(
            f"{store.uri(key)} has SHA-256 {actual}, but production.json ({pointer.model_version}) "
            f"records {pointer.bundle_sha256}: refusing to unpickle a modified or corrupted bundle"
        )
    bundle = deserialize_bundle(data)
    if bundle.model_version != pointer.model_version:
        raise ValueError(
            f"production.json references {pointer.model_version} but its bundle is {bundle.model_version}"
        )
    return bundle


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
    """The facts of a past training run that --skip-if-unchanged needs.

    ``data_watermark_us`` (end of the newest completed lifecycle the run saw, from
    params_json) is None for runs recorded before it existed; ``lifecycles`` is then
    the fallback signal.
    """

    model_version: str
    lifecycles: int
    promoted: bool = False
    data_watermark_us: int | None = None


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
            "SELECT model_version, train_lifecycles + test_lifecycles, promoted, "
            "JSONExtract(params_json, 'data_watermark_us', 'Nullable(Int64)') "
            f"FROM {ML_MODEL_RUNS_TABLE} ORDER BY trained_at DESC, model_version DESC LIMIT 1"
        )
        if not rows:
            return None
        version, lifecycles, promoted, watermark = rows[0]
        return RunSummary(
            model_version=str(version),
            lifecycles=int(lifecycles),
            promoted=bool(promoted),
            data_watermark_us=None if watermark is None else int(watermark),
        )

    def record_run(self, record: ModelRunRecord) -> None:
        """Insert one row into ml_model_runs."""
        self._gateway.insert_rows(ML_MODEL_RUNS_TABLE, [record.as_row()], ML_MODEL_RUNS_COLUMNS)
        logger.info("recorded run %s in %s", record.model_version, ML_MODEL_RUNS_TABLE)
