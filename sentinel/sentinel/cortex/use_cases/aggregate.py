# Copyright 2026 Alexandre Cardoso
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from sentinel.domain.models import ModelPhase, TemporalAnomalySignal, TemporalHeartbeatEvent


@dataclass
class LayerStateVector:
    layer_name: str
    last_seen: datetime
    model_phase: ModelPhase = ModelPhase.WARMUP
    anomaly_rate: float = 0.0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    anomaly_rate_z: float = 0.0
    volume_z: float = 0.0
    bucket_confidence: int = 0


def update_layer_state(
    layer_vectors: dict[str, LayerStateVector],
    event: TemporalHeartbeatEvent | TemporalAnomalySignal,
) -> dict[str, LayerStateVector]:
    """Update the state vector for a temporal layer from an incoming event.

    Heartbeats update liveness and operational metrics.
    AnomalySignals additionally update z-score fields.
    """
    if isinstance(event, TemporalHeartbeatEvent):
        existing = layer_vectors.get(event.layer_name)
        layer_vectors[event.layer_name] = LayerStateVector(
            layer_name=event.layer_name,
            last_seen=event.timestamp,
            model_phase=event.model_phase,
            anomaly_rate=event.anomaly_rate,
            avg_score=event.avg_score,
            avg_latency_ms=event.avg_latency_ms,
            anomaly_rate_z=existing.anomaly_rate_z if existing else 0.0,
            volume_z=existing.volume_z if existing else 0.0,
            bucket_confidence=event.bucket_confidence,
        )
    elif isinstance(event, TemporalAnomalySignal):
        existing = layer_vectors.get(event.layer_name)
        layer_vectors[event.layer_name] = LayerStateVector(
            layer_name=event.layer_name,
            last_seen=event.window_end,
            model_phase=existing.model_phase if existing else ModelPhase.INFERENCE,
            anomaly_rate=event.anomaly_rate,
            avg_score=existing.avg_score if existing else 0.0,
            avg_latency_ms=existing.avg_latency_ms if existing else 0.0,
            anomaly_rate_z=event.anomaly_rate_z,
            volume_z=event.volume_z,
            bucket_confidence=event.bucket_confidence,
        )

    return layer_vectors


def check_layer_silence(
    layer_vectors: dict[str, LayerStateVector],
    layer_name: str,
    threshold_s: float,
    now: datetime,
) -> bool:
    """Return True if the temporal layer has been silent for longer than threshold_s."""
    vector = layer_vectors.get(layer_name)
    if vector is None:
        return False
    elapsed = (now - vector.last_seen).total_seconds()
    return elapsed > threshold_s


def build_layer_feature_matrix(
    layer_vectors: dict[str, LayerStateVector],
) -> np.ndarray:
    """Build a 2D numpy array of features from all layer state vectors.

    Features per row: [anomaly_rate, avg_score, avg_latency_ms, anomaly_rate_z, volume_z]
    """
    if not layer_vectors:
        return np.empty((0, 5))

    rows = [
        [
            lv.anomaly_rate,
            lv.avg_score,
            lv.avg_latency_ms,
            lv.anomaly_rate_z,
            lv.volume_z,
        ]
        for lv in layer_vectors.values()
    ]
    return np.array(rows, dtype=float)
