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
from enum import Enum


class ModelPhase(str, Enum):
    TRAINING = "TRAINING"
    INFERENCE = "INFERENCE"
    COLD = "COLD"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    CAUSAL = "CAUSAL"
    SYSTEMIC = "SYSTEMIC"
    SILENCE = "SILENCE"
    TIMEOUT = "TIMEOUT"


class AnomalyType(str, Enum):
    TIMEOUT = "TIMEOUT"
    SCORE = "SCORE"
    RULE = "RULE"


@dataclass
class RawMessage:
    id: str
    body: dict
    received_at: datetime
    size_bytes: int = 0


@dataclass
class ProcessedEvent:
    agent_name: str
    correlation_id: str
    input_received_at: datetime
    output_sent_at: datetime
    processing_latency_ms: float
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: AnomalyType | None
    payload_schema_hash: str
    input_size_bytes: int
    output_size_bytes: int
    input_body: dict | None = None
    output_body: dict | None = None


@dataclass
class HeartbeatEvent:
    agent_name: str
    timestamp: datetime
    message_rate_per_sec: float
    error_rate_last_window: float
    last_message_timestamp: datetime
    model_phase: ModelPhase


@dataclass
class Alert:
    source: str
    severity: AlertSeverity
    alert_type: AlertType
    affected_agents: list[str]
    description: str
    timestamp: datetime
    diagnosis: dict = field(default_factory=dict)


@dataclass
class InputCapture:
    """One input stream's contribution to a correlated pair."""
    name: str
    body: dict
    received_at: datetime
    size_bytes: int


@dataclass
class CorrelatedPair:
    """Product of the CorrelationEngine: N matched inputs + one output, ready for detection."""
    engine_name: str
    correlation_id: str
    inputs: list[InputCapture]
    output_body: dict
    output_received_at: datetime
    output_size_bytes: int
    processing_latency_ms: float
    timed_out: bool = False

    @property
    def input_received_at(self) -> datetime:
        """Earliest input arrival time (or output time on timeout)."""
        if not self.inputs:
            return self.output_received_at
        return min(i.received_at for i in self.inputs)

    @property
    def input_size_bytes(self) -> int:
        return sum(i.size_bytes for i in self.inputs)

    @property
    def input_body(self) -> dict:
        """First input body for single-input engines; name-keyed dict for multi-input."""
        if not self.inputs:
            return {}
        if len(self.inputs) == 1:
            return self.inputs[0].body
        return {inp.name: inp.body for inp in self.inputs}


@dataclass
class VersionMeta:
    version_id: str
    created_at: datetime
    performance_score: float | None = None
