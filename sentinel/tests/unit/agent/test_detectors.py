"""Unit tests for detector implementations and the factory."""

import math

import pytest

from sentinel.agent.detectors.factory import create_detector
from sentinel.agent.detectors.isolation_forest_detector import IsolationForestDetector
from sentinel.agent.detectors.nri_detector import NRIDetector
from sentinel.config.schema import DetectorConfig, FieldMapConfig


# ── IsolationForestDetector ───────────────────────────────────────────────────

def _if_detector(threshold=-0.1):
    cfg = DetectorConfig(
        name="if",
        algorithm="isolation_forest",
        anomaly_threshold=threshold,
        n_estimators=10,
        training_window=20,
    )
    return IsolationForestDetector(cfg)


def _operational_event(latency=100.0, in_size=200, out_size=300, schema_hash="abc"):
    return {
        "processing_latency_ms": latency,
        "input_size_bytes": in_size,
        "output_size_bytes": out_size,
        "payload_schema_hash": schema_hash,
        "_input_body": {},
        "_output_body": {},
    }


def test_if_fit_and_score_normal():
    det = _if_detector()
    samples = [_operational_event(latency=100 + i) for i in range(30)]
    model = det.fit(samples)
    assert model is not None

    score = det.score(model, _operational_event(latency=110.0))
    assert isinstance(score, float)


def test_if_is_anomaly_direction():
    det = _if_detector(threshold=-0.1)
    # IF uses score < threshold for anomaly
    assert det.is_anomaly(-0.5) is True
    assert det.is_anomaly(0.0) is False
    assert det.is_anomaly(-0.05) is False


def test_if_extract_features():
    det = _if_detector()
    features = det.extract_features(_operational_event(latency=50.0, in_size=100, out_size=200))
    assert len(features) == 4
    assert features[0] == 50.0  # latency
    assert features[1] == 100   # input_size
    assert features[2] == 200   # output_size


# ── NRIDetector ───────────────────────────────────────────────────────────────

def test_nri_fit_raises_not_implemented():
    cfg = DetectorConfig(name="nri", algorithm="nri")
    det = NRIDetector(cfg)
    with pytest.raises(NotImplementedError):
        det.fit([{"a": 1}])


def test_nri_score_returns_zero():
    cfg = DetectorConfig(name="nri", algorithm="nri")
    det = NRIDetector(cfg)
    assert det.score(None, {"x": 1.0}) == 0.0


def test_nri_is_anomaly_always_false():
    cfg = DetectorConfig(name="nri", algorithm="nri")
    det = NRIDetector(cfg)
    assert det.is_anomaly(99999.0) is False


# ── Factory ───────────────────────────────────────────────────────────────────

def test_factory_creates_isolation_forest():
    cfg = DetectorConfig(name="if", algorithm="isolation_forest")
    det = create_detector(cfg)
    assert isinstance(det, IsolationForestDetector)


def test_factory_creates_nri():
    cfg = DetectorConfig(name="nri", algorithm="nri")
    det = create_detector(cfg)
    assert isinstance(det, NRIDetector)


def test_factory_unknown_algorithm_raises():
    cfg = DetectorConfig(name="unknown", algorithm="isolation_forest")
    # Override algorithm attr post-construction to simulate unknown
    cfg.__dict__["algorithm"] = "unknown_algo"
    with pytest.raises(ValueError, match="unknown"):
        create_detector(cfg)


# ── cVAE requires field_map ───────────────────────────────────────────────────

def test_cvae_requires_field_map():
    from sentinel.agent.detectors.cvae_detector import ConditionalVAEDetector
    cfg = DetectorConfig(name="cvae", algorithm="cvae", field_map=None)
    with pytest.raises(ValueError, match="field_map"):
        ConditionalVAEDetector(cfg)


def test_cvae_fit_and_score():
    pytest.importorskip("torch")
    from sentinel.agent.detectors.cvae_detector import ConditionalVAEDetector

    cfg = DetectorConfig(
        name="cvae",
        algorithm="cvae",
        field_map=FieldMapConfig(input=["a", "b"], output=["c"]),
        hidden_dim=8,
        latent_dim=4,
        learning_rate=0.01,
        epochs=3,
        training_window=10,
        anomaly_threshold=10.0,
    )
    det = ConditionalVAEDetector(cfg)

    def _sample(a, b, c):
        return {
            "_input_body": {"a": a, "b": b},
            "_output_body": {"c": c},
            "processing_latency_ms": 10.0,
            "input_size_bytes": 50,
            "output_size_bytes": 50,
            "payload_schema_hash": "abc",
        }

    samples = [_sample(1.0, 2.0, 3.0) for _ in range(20)]
    model = det.fit(samples)
    assert model is not None

    score = det.score(model, _sample(1.0, 2.0, 3.0))
    assert isinstance(score, float)
    assert score >= 0.0  # MSE is non-negative


def test_cvae_anomaly_direction():
    pytest.importorskip("torch")
    from sentinel.agent.detectors.cvae_detector import ConditionalVAEDetector

    cfg = DetectorConfig(
        name="cvae",
        algorithm="cvae",
        field_map=FieldMapConfig(input=["a"], output=["b"]),
        anomaly_threshold=1.0,
    )
    det = ConditionalVAEDetector(cfg)
    # cVAE uses score > threshold for anomaly
    assert det.is_anomaly(2.0) is True
    assert det.is_anomaly(0.5) is False


# ── MAFDetector ───────────────────────────────────────────────────────────────

def test_maf_fit_and_score():
    pytest.importorskip("torch")
    from sentinel.agent.detectors.maf_detector import MAFDetector

    cfg = DetectorConfig(
        name="maf",
        algorithm="maf",
        hidden_dim=16,
        learning_rate=0.01,
        epochs=3,
        training_window=10,
        anomaly_threshold=10.0,
    )
    det = MAFDetector(cfg)

    def _sample(x, y):
        return {
            "_input_body": {"x": x},
            "_output_body": {"y": y},
            "processing_latency_ms": 10.0,
            "input_size_bytes": 50,
            "output_size_bytes": 50,
            "payload_schema_hash": "abc",
        }

    samples = [_sample(float(i), float(i * 2)) for i in range(20)]
    model = det.fit(samples)
    assert model is not None

    score = det.score(model, _sample(5.0, 10.0))
    assert isinstance(score, float)
    # -log_likelihood can be any real number; just check it's finite
    assert math.isfinite(score)


def test_maf_anomaly_direction():
    pytest.importorskip("torch")
    from sentinel.agent.detectors.maf_detector import MAFDetector

    cfg = DetectorConfig(
        name="maf", algorithm="maf", anomaly_threshold=5.0
    )
    det = MAFDetector(cfg)
    assert det.is_anomaly(10.0) is True
    assert det.is_anomaly(1.0) is False
