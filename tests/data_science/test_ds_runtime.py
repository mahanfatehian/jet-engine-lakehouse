"""Exit-code convention, bounded retries, transient-error classification and settings."""

from __future__ import annotations

import pytest
from artifact_store import _is_transient_s3_error
from botocore.exceptions import ClientError, EndpointConnectionError
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from ds_runtime import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_SKIPPED,
    PreconditionsNotMetError,
    StopToken,
    TerminationRequestedError,
    retry_call,
    run_cli,
)
from ds_settings import ConfigError, PromotionSettings, load_settings
from lakehouse_io import _is_transient, final_modifier


def _raise(exc: Exception) -> int:
    raise exc


def test_exit_codes_follow_the_contract() -> None:
    assert (EXIT_OK, EXIT_FAILURE, EXIT_SKIPPED, EXIT_INTERRUPTED) == (0, 1, 3, 143)


@pytest.mark.parametrize(
    ("main", "code"),
    [
        (lambda: EXIT_OK, EXIT_OK),
        (lambda: _raise(PreconditionsNotMetError("no data yet")), EXIT_SKIPPED),
        (lambda: _raise(TerminationRequestedError("stop")), EXIT_INTERRUPTED),
        (lambda: _raise(RuntimeError("boom")), EXIT_FAILURE),
    ],
)
def test_run_cli_maps_outcomes_to_exit_codes(main, code: int) -> None:
    with pytest.raises(SystemExit) as exited:
        run_cli(main)

    assert exited.value.code == code


def test_retry_call_retries_transient_errors_a_bounded_number_of_times() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "ok"

    assert retry_call(flaky, description="flaky", is_retryable=lambda _: True, base_delay_seconds=0.0) == "ok"
    assert len(calls) == 3

    calls.clear()
    with pytest.raises(ConnectionError):
        retry_call(
            flaky, description="flaky", is_retryable=lambda _: True, attempts=2, base_delay_seconds=0.0
        )
    assert len(calls) == 2


def test_retry_call_does_not_retry_permanent_errors() -> None:
    calls: list[int] = []

    def broken() -> None:
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        retry_call(broken, description="broken", is_retryable=lambda _: False, base_delay_seconds=0.0)
    assert len(calls) == 1


def test_transient_error_classification() -> None:
    assert _is_transient(OperationalError("connection refused"))
    assert _is_transient(DatabaseError("too many queries", code=202))
    assert not _is_transient(DatabaseError("unknown table", code=60))
    assert _is_transient_s3_error(EndpointConnectionError(endpoint_url="http://minio:9000"))
    slow_down = {"Error": {"Code": "SlowDown"}, "ResponseMetadata": {"HTTPStatusCode": 503}}
    assert _is_transient_s3_error(ClientError(slow_down, "PutObject"))
    denied = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}
    assert not _is_transient_s3_error(ClientError(denied, "PutObject"))


def test_final_is_only_applied_to_engines_that_merge_versions() -> None:
    assert final_modifier("ReplacingMergeTree") == "FINAL"
    assert final_modifier("ReplicatedReplacingMergeTree") == "FINAL"
    assert final_modifier("MergeTree") == ""
    assert final_modifier("View") == ""


def test_stop_token_wakes_waiters_and_raises_at_safe_points() -> None:
    stop = StopToken()
    assert not stop.wait(0.01)
    stop.raise_if_stopped()

    stop.request_stop()

    assert stop.stop_requested and stop.wait(10)
    with pytest.raises(TerminationRequestedError):
        stop.raise_if_stopped()


def test_settings_defaults_follow_the_contract() -> None:
    settings = load_settings({})

    assert (settings.clickhouse.host, settings.clickhouse.port) == ("localhost", 8123)
    assert (settings.clickhouse.user, settings.clickhouse.database) == ("clickhouse", "iot_engine")
    assert settings.s3.endpoint_url == "http://localhost:9000"
    assert (settings.s3.bucket, settings.s3.region) == ("lakehouse", "us-east-1")
    model = settings.model
    assert (model.failure_horizon_cycles, model.rul_cap_cycles, model.min_training_lifecycles) == (25, 125, 6)
    assert (model.flight_hours_per_cycle, model.promotion_policy) == (2.0, "better")
    assert model.artifact_prefix == "ml-artifacts/rul_model"
    assert model.training_max_rows == 600_000
    assert (model.promotion.min_r2, model.promotion.min_roc_auc, model.promotion.f1_tolerance) == (
        0.5,
        0.75,
        0.05,
    )
    assert settings.snapshot_retention == 5
    assert settings.snapshot_window_days == 7
    assert model.spectral_max_lag_seconds == 60.0
    assert "password" not in repr(settings.clickhouse)
    assert "secret_access_key" not in repr(settings.s3)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RUL_CAP_CYCLES": "abc"}, "integer"),
        ({"MIN_TRAINING_LIFECYCLES": "0"}, ">= 1"),
        ({"FLIGHT_HOURS_PER_CYCLE": "-1"}, "> 0"),
        ({"MODEL_PROMOTION_POLICY": "sometimes"}, "better, always"),
        ({"RUL_CAP_CYCLES": "20", "RUL_FAILURE_HORIZON_CYCLES": "25"}, "must exceed"),
        ({"ML_ARTIFACT_PREFIX": "///"}, "must not be empty"),
        ({"TRAINING_MAX_ROWS": "999"}, ">= 1000"),
        ({"SNAPSHOT_WINDOW_DAYS": "0"}, ">= 1"),
        ({"PROMOTION_MIN_R2": "1.5"}, r"PROMOTION_MIN_R2 must be a finite number in \[-inf, 1\]"),
        ({"PROMOTION_MIN_R2": "-inf"}, "finite"),
        ({"PROMOTION_MIN_ROC_AUC": "nan"}, "PROMOTION_MIN_ROC_AUC"),
        ({"PROMOTION_MIN_ROC_AUC": "high"}, "must be a number"),
        ({"PROMOTION_F1_TOLERANCE": "-0.1"}, r"\[0, 1\]"),
        (
            {"SPECTRAL_MAX_LAG_SECONDS": "-1"},
            r"SPECTRAL_MAX_LAG_SECONDS must be a finite number in \[0, 3600\]",
        ),
        ({"SPECTRAL_MAX_LAG_SECONDS": "7200"}, r"\[0, 3600\]"),
        ({"SPECTRAL_MAX_LAG_SECONDS": "nan"}, "SPECTRAL_MAX_LAG_SECONDS"),
    ],
)
def test_invalid_settings_are_rejected(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_settings(env)


def test_promotion_and_window_settings_are_read_from_the_environment() -> None:
    settings = load_settings(
        {
            "TRAINING_MAX_ROWS": "50000",
            "SNAPSHOT_WINDOW_DAYS": "2",
            "PROMOTION_MIN_R2": "-0.5",
            "PROMOTION_MIN_ROC_AUC": "0.9",
            "PROMOTION_F1_TOLERANCE": "0",
            "SPECTRAL_MAX_LAG_SECONDS": "0",
        }
    )

    assert settings.model.training_max_rows == 50_000 and settings.snapshot_window_days == 2
    assert settings.model.promotion == PromotionSettings(min_r2=-0.5, min_roc_auc=0.9, f1_tolerance=0.0)
    assert settings.model.spectral_max_lag_seconds == 0.0
    assert load_settings({"SPECTRAL_MAX_LAG_SECONDS": "120"}).model.spectral_max_lag_seconds == 120.0
