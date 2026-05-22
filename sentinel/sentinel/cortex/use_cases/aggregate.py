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

from sentinel.domain.models import HeartbeatEvent, ModelPhase, ProcessedEvent


@dataclass
class AgentStateVector:
    agent_name: str
    last_seen: datetime
    message_rate: float = 0.0
    error_rate: float = 0.0
    anomaly_score: float = 0.0
    model_phase: ModelPhase = ModelPhase.COLD


def update_state(
    vectors: dict[str, AgentStateVector],
    event: ProcessedEvent | HeartbeatEvent,
) -> dict[str, AgentStateVector]:
    """Update the state vector for an agent from an incoming event or heartbeat.

    Mutates and returns the same dict — callers own the dict and always reassign,
    so there is no need to copy N entries when only one changes.
    """
    if isinstance(event, ProcessedEvent):
        agent_name = event.agent_name
        existing = vectors.get(agent_name)
        vectors[agent_name] = AgentStateVector(
            agent_name=agent_name,
            last_seen=event.output_sent_at,
            message_rate=existing.message_rate if existing else 0.0,
            error_rate=existing.error_rate if existing else 0.0,
            anomaly_score=event.anomaly_score,
            model_phase=existing.model_phase if existing else ModelPhase.COLD,
        )
    elif isinstance(event, HeartbeatEvent):
        agent_name = event.agent_name
        existing = vectors.get(agent_name)
        vectors[agent_name] = AgentStateVector(
            agent_name=agent_name,
            last_seen=event.timestamp,
            message_rate=event.message_rate_per_sec,
            error_rate=event.error_rate_last_window,
            anomaly_score=existing.anomaly_score if existing else 0.0,
            model_phase=event.model_phase,
        )

    return vectors


def check_silence(
    vectors: dict[str, AgentStateVector],
    agent_name: str,
    threshold_s: float,
    now: datetime,
) -> bool:
    """Return True if the agent has been silent for longer than threshold_s."""
    vector = vectors.get(agent_name)
    if vector is None:
        return False
    elapsed = (now - vector.last_seen).total_seconds()
    return elapsed > threshold_s


def build_feature_matrix(vectors: dict[str, AgentStateVector]) -> np.ndarray:
    """Build a 2D numpy array of features from all agent state vectors."""
    if not vectors:
        return np.empty((0, 4))

    rows = []
    for sv in vectors.values():
        rows.append(
            [
                sv.message_rate,
                sv.error_rate,
                sv.anomaly_score,
                float(sv.model_phase == ModelPhase.INFERENCE),
            ]
        )
    return np.array(rows, dtype=float)
