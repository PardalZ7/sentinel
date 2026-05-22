import pytest

from sentinel.agent.isolation_forest import (
    create_model,
    extract_features,
    fit_model,
    is_anomaly,
    score_event,
)
from sentinel.config.schema import ModelIsolationForestConfig


def _default_config() -> ModelIsolationForestConfig:
    return ModelIsolationForestConfig(
        n_estimators=10,
        contamination="auto",
        anomaly_threshold=-0.1,
        training_window=20,
    )


def _sample_event(latency=100.0, input_size=512, output_size=256, schema_hash="abc123"):
    return {
        "processing_latency_ms": latency,
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "payload_schema_hash": schema_hash,
    }


# ── extract_features ──────────────────────────────────────────────────────────

def test_extract_features_returns_4_values():
    event = _sample_event()
    features = extract_features(event)
    assert len(features) == 4


def test_extract_features_values():
    event = _sample_event(latency=200.0, input_size=1024, output_size=512)
    features = extract_features(event)
    assert features[0] == 200.0
    assert features[1] == 1024.0
    assert features[2] == 512.0
    assert isinstance(features[3], float)


def test_extract_features_empty_schema_hash():
    event = _sample_event(schema_hash="")
    features = extract_features(event)
    assert features[3] == 0.0


def test_extract_features_missing_keys():
    event = {}
    features = extract_features(event)
    assert features == [0.0, 0.0, 0.0, 0.0]


# ── create_model ──────────────────────────────────────────────────────────────

def test_create_model():
    config = _default_config()
    model = create_model(config)
    assert model is not None
    assert model.n_estimators == 10


# ── fit_model ─────────────────────────────────────────────────────────────────

def test_fit_model():
    config = _default_config()
    model = create_model(config)
    samples = [_sample_event(latency=float(i * 10)) for i in range(1, 25)]
    fitted = fit_model(model, samples)
    # After fitting, the model should have the estimators_ attribute
    assert hasattr(fitted, "estimators_")


# ── score_event ───────────────────────────────────────────────────────────────

def test_score_event_returns_float():
    config = _default_config()
    model = create_model(config)
    samples = [_sample_event(latency=float(i * 5)) for i in range(1, 25)]
    fitted = fit_model(model, samples)
    event = _sample_event(latency=100.0)
    score = score_event(fitted, event)
    assert isinstance(score, float)


def test_score_event_extreme_outlier_is_lower():
    """An extreme outlier should score lower (more negative) than normal data."""
    config = _default_config()
    model = create_model(config)
    # Train on very consistent data
    normal_samples = [_sample_event(latency=100.0) for _ in range(30)]
    fitted = fit_model(model, normal_samples)

    normal_score = score_event(fitted, _sample_event(latency=100.0))
    outlier_score = score_event(fitted, _sample_event(latency=1_000_000.0))

    assert outlier_score <= normal_score


# ── is_anomaly ────────────────────────────────────────────────────────────────

def test_is_anomaly_below_threshold():
    assert is_anomaly(-0.5, threshold=-0.1) is True


def test_is_anomaly_above_threshold():
    assert is_anomaly(0.2, threshold=-0.1) is False


def test_is_anomaly_equal_threshold():
    assert is_anomaly(-0.1, threshold=-0.1) is False
