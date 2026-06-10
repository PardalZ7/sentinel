"""Unit tests for the VAE, SVDD, LSTM-AE and TCN-AE detectors.

PyTorch-dependent: skipped entirely when torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from sentinel.agent.detectors.factory import create_detector
from sentinel.agent.detectors.lstm_ae_detector import LSTMAEDetector
from sentinel.agent.detectors.svdd_detector import SVDDDetector
from sentinel.agent.detectors.tcn_ae_detector import TCNAEDetector
from sentinel.agent.detectors.vae_detector import VAEDetector
from sentinel.config.schema import DetectorConfig, FieldMapConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _event(latency=100.0, in_size=200, out_size=300, a=1.0, b=2.0):
    return {
        "processing_latency_ms": latency,
        "input_size_bytes": in_size,
        "output_size_bytes": out_size,
        "payload_schema_hash": "abc",
        "_input_body": {"a": a, "b": b},
        "_output_body": {"sum": a + b},
    }


def _normal_samples(n=60):
    # Mild noise around a stable operating point.
    return [
        _event(latency=100.0 + (i % 7), a=1.0 + 0.01 * (i % 5), b=2.0 + 0.01 * (i % 3))
        for i in range(n)
    ]


def _fast_cfg(name, algorithm, **overrides):
    defaults = dict(
        name=name,
        algorithm=algorithm,
        anomaly_threshold=0.05,
        training_window=50,
        test_buffer_size=20,
        hidden_dim=8,
        latent_dim=4,
        learning_rate=0.01,
        epochs=5,
        sequence_length=5,
    )
    defaults.update(overrides)
    return DetectorConfig(**defaults)


def _window(events):
    return {"_sequence": events}


# ── VAE ───────────────────────────────────────────────────────────────────────

def test_vae_fit_score_and_threshold():
    det = VAEDetector(_fast_cfg("vae", "vae"))
    model = det.fit(_normal_samples())

    assert "auto_threshold" in model
    assert model["fields"] == ["a", "b", "sum"]   # auto-discovered, sorted

    score_normal = det.score(model, _event())
    assert isinstance(score_normal, float)

    # An extreme outlier must score worse than a normal event.
    score_outlier = det.score(model, _event(a=10_000.0, b=-5_000.0))
    assert score_outlier > score_normal
    assert det.is_anomaly(score_outlier, model) is True


def test_vae_explicit_field_map():
    cfg = _fast_cfg("vae", "vae", field_map=FieldMapConfig(input=["a"], output=["sum"]))
    det = VAEDetector(cfg)
    model = det.fit(_normal_samples())
    assert model["fields"] == ["a", "sum"]


def test_vae_batch_score_matches_score():
    det = VAEDetector(_fast_cfg("vae", "vae"))
    model = det.fit(_normal_samples())
    events = [_event(), _event(a=50.0)]
    batch = det.batch_score(model, events)
    assert len(batch) == 2
    assert batch[0] == pytest.approx(det.score(model, events[0]), abs=1e-5)


def test_vae_serialization_round_trip():
    det = VAEDetector(_fast_cfg("vae", "vae"))
    model = det.fit(_normal_samples())
    # Simulate disk round-trip: a fresh dict identity forces cache rebuild from state_dict.
    reloaded = {**model}
    assert det.score(reloaded, _event()) == pytest.approx(det.score(model, _event()), abs=1e-5)


# ── SVDD ──────────────────────────────────────────────────────────────────────

def test_svdd_fit_score_and_threshold():
    det = SVDDDetector(_fast_cfg("svdd", "svdd", nu=0.1))
    model = det.fit(_normal_samples())

    assert "auto_threshold" in model
    assert "center" in model

    score_normal = det.score(model, _event())
    score_outlier = det.score(model, _event(a=10_000.0, b=-5_000.0))
    assert score_outlier > score_normal
    assert det.is_anomaly(score_outlier, model) is True


def test_svdd_threshold_respects_nu():
    det = SVDDDetector(_fast_cfg("svdd", "svdd", nu=0.1))
    model = det.fit(_normal_samples())
    # By construction ~nu of training samples sit above the (1-nu) percentile.
    scores = det.batch_score(model, _normal_samples())
    flagged = sum(1 for s in scores if det.is_anomaly(s, model))
    assert flagged / len(scores) <= 0.25


def test_svdd_batch_score():
    det = SVDDDetector(_fast_cfg("svdd", "svdd"))
    model = det.fit(_normal_samples())
    batch = det.batch_score(model, [_event(), _event()])
    assert len(batch) == 2
    assert all(isinstance(s, float) for s in batch)


# ── Sequence detectors (shared behaviour) ─────────────────────────────────────

@pytest.mark.parametrize("cls,algo", [(LSTMAEDetector, "lstm_ae"), (TCNAEDetector, "tcn_ae")])
def test_sequence_detector_properties(cls, algo):
    det = cls(_fast_cfg(algo, algo))
    assert det.requires_sequence is True
    assert det.sequence_length == 5
    assert det.algorithm == algo


@pytest.mark.parametrize("cls,algo", [(LSTMAEDetector, "lstm_ae"), (TCNAEDetector, "tcn_ae")])
def test_sequence_fit_and_score(cls, algo):
    det = cls(_fast_cfg(algo, algo))
    samples = _normal_samples(40)
    model = det.fit(samples)

    assert "auto_threshold" in model
    assert model["sequence_length"] == 5

    normal_window = _window(samples[:5])
    score_normal = det.score(model, normal_window)
    assert isinstance(score_normal, float)

    # A window with an extreme latency spike reconstructs worse.
    spike = [_event(latency=100.0)] * 4 + [_event(latency=1_000_000.0)]
    score_spike = det.score(model, _window(spike))
    assert score_spike > score_normal


@pytest.mark.parametrize("cls,algo", [(LSTMAEDetector, "lstm_ae"), (TCNAEDetector, "tcn_ae")])
def test_sequence_short_window_is_padded(cls, algo):
    det = cls(_fast_cfg(algo, algo))
    model = det.fit(_normal_samples(40))
    # Short window must not raise — it is left-padded with zeros.
    score = det.score(model, _window([_event(), _event()]))
    assert isinstance(score, float)


@pytest.mark.parametrize("cls,algo", [(LSTMAEDetector, "lstm_ae"), (TCNAEDetector, "tcn_ae")])
def test_sequence_fit_requires_enough_samples(cls, algo):
    det = cls(_fast_cfg(algo, algo))
    with pytest.raises(ValueError, match="sequence_length"):
        det.fit(_normal_samples(3))


@pytest.mark.parametrize("cls,algo", [(LSTMAEDetector, "lstm_ae"), (TCNAEDetector, "tcn_ae")])
def test_sequence_batch_score(cls, algo):
    det = cls(_fast_cfg(algo, algo))
    samples = _normal_samples(40)
    model = det.fit(samples)
    windows = [_window(samples[:5]), _window(samples[5:10])]
    batch = det.batch_score(model, windows)
    assert len(batch) == 2
    assert batch[0] == pytest.approx(det.score(model, windows[0]), abs=1e-5)


def test_sequence_extract_features_is_flat():
    det = LSTMAEDetector(_fast_cfg("lstm", "lstm_ae"))
    features = det.extract_features(_window([_event()] * 5))
    assert len(features) == 5 * 3   # sequence_length × features_per_step


# ── Factory ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("algo,cls", [
    ("vae", VAEDetector),
    ("svdd", SVDDDetector),
    ("lstm_ae", LSTMAEDetector),
    ("tcn_ae", TCNAEDetector),
])
def test_factory_creates_new_detectors(algo, cls):
    det = create_detector(_fast_cfg(algo, algo))
    assert isinstance(det, cls)
