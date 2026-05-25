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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    password: str | None = None


class AwsConfig(BaseModel):
    region: str = "us-east-1"
    endpoint_url: str | None = None   # set to http://localhost:4566 for LocalStack


class DashboardConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8888


class ControlPlaneConfig(BaseModel):
    enabled: bool = False


class TransportResourceConfig(BaseModel):
    type: Literal["sqs", "sns", "kafka"] = "sqs"
    resource: str = ""


class ModelIsolationForestConfig(BaseModel):
    n_estimators: int = 100
    contamination: str | float = "auto"
    anomaly_threshold: float = -0.1
    training_window: int = 500
    test_buffer_size: int = 100


class ModelAutoencoderConfig(BaseModel):
    hidden_dim: int = 64
    training_window: int = 1000
    learning_rate: float = 0.001
    epochs: int = 50
    test_buffer_size: int = 200


class ModelConfig(BaseModel):
    max_versions: int = 5
    auto_rollback: bool = False
    isolation_forest: ModelIsolationForestConfig = Field(
        default_factory=ModelIsolationForestConfig
    )
    autoencoder: ModelAutoencoderConfig = Field(default_factory=ModelAutoencoderConfig)


class ReportingConfig(BaseModel):
    heartbeat_interval_s: int = 30
    send_all_events: bool = False


class CortexRef(BaseModel):
    host: str = "localhost"
    port: int = 50051


class FieldMapConfig(BaseModel):
    """Payload field mapping for semantic detectors (cVAE, MAF, NRI).

    Specifies which fields from the observed application's input and output
    payloads the detector should use for semantic anomaly detection.

    This configuration belongs in the usecase (sentinel.json), NOT in the
    Sentinel core — field names are application-specific knowledge.

    Example: if the observed application receives {a, b} and produces {a, b, sum},
    the field_map would be:
      input:  ["a", "b"]
      output: ["sum"]

    Fields are extracted via exact key lookup from the raw message body.
    Nested fields are NOT supported — flatten payloads before monitoring if needed.
    Missing fields default to 0.0 to ensure resilience against partial payloads.
    """

    input: list[str] = Field(default_factory=list)
    output: list[str] = Field(default_factory=list)


class DetectorConfig(BaseModel):
    """Configuration for a single anomaly detector within an agent.

    An agent may have multiple detectors, each running a different algorithm
    on the same correlated event. Each detector has its own training lifecycle,
    champion/challenger cycle, and phase state.

    Algorithm options:
      isolation_forest  — Operational anomaly detection (latency, sizes, schema).
                          Fast to train (~500 samples). No field_map needed.
      cvae              — Conditional VAE: learns P(output_fields | input_fields).
                          Detects when output values are semantically inconsistent
                          with the given inputs. Requires field_map.
      maf               — Masked Autoregressive Flow (MADE): learns the joint
                          distribution P(all_fields). Scores via -log_likelihood.
                          More data-hungry (~2000+ samples). field_map optional.
      nri               — Neural Relational Inference: learns field dependency graphs.
                          Currently a stub — interface is stable for future implementation.

    Threshold semantics differ by algorithm family:
      isolation_forest  — anomaly when score < threshold  (threshold is negative, e.g. -0.1)
      cvae / maf / nri  — anomaly when score > threshold  (threshold is positive, e.g. 0.05)

    The field_map is only meaningful for semantic detectors (cvae, maf, nri).
    It should be configured in the usecase's sentinel.json, not in library defaults,
    because field names are application-specific knowledge unknown to Sentinel.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    algorithm: Literal["isolation_forest", "cvae", "maf", "nri"] = "isolation_forest"
    phase: str = "TRAINING"
    training_window: int = 500
    test_buffer_size: int = 100
    anomaly_threshold: float = -0.1
    field_map: FieldMapConfig | None = None

    # IsolationForest-specific parameters (ignored by other algorithms)
    n_estimators: int = 100
    contamination: str | float = "auto"

    # Neural network parameters (cVAE and MAF; ignored by isolation_forest)
    hidden_dim: int = 32
    latent_dim: int = 8
    learning_rate: float = 0.001
    epochs: int = 30

    # Auto-inference: when set, the detector auto-transitions TRAINING→INFERENCE
    # once the champion FP rate drops below this threshold (0.0–1.0). None = disabled.
    auto_infer_fp_threshold: float | None = None

    # Gating detector: name of another detector on the same agent whose model is used
    # to filter contaminated training samples. Samples flagged as anomalous by the gate
    # are discarded before entering this detector's training buffer. Requires the gate
    # detector to already have a champion model (e.g. isolation_forest converges first).
    gating_detector: str | None = None

    # Max consecutive challenger rejections before stopping training and emitting a
    # degraded warning. 0 = unlimited (trains forever). Prevents infinite CPU burn
    # when the model cannot converge (e.g. training data is too noisy).
    max_challengers_rejected: int = 0


class CorrelationEngineConfig(BaseModel):
    """Configuration for a CorrelationEngine instance running on the use-case side.

    The engine consumes input and output queues, correlates messages via Redis,
    and publishes CorrelatedPair objects to one or more destination queues (agents).

    On timeout (output arrives but no matching input found), a notification is sent
    to timeout_topic_arn if configured; otherwise the event is logged and dropped.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    input: TransportResourceConfig = Field(default_factory=TransportResourceConfig)
    output: TransportResourceConfig = Field(default_factory=TransportResourceConfig)
    correlation_field: str = "correlationId"
    input_correlation_field: str | None = None
    output_correlation_field: str | None = None
    correlation_mode: Literal["normal", "grouping", "splitting"] = "normal"
    correlation_ttl_s: int = 300
    destinations: list[TransportResourceConfig] = Field(default_factory=list)
    timeout_topic_arn: str | None = None

    @property
    def effective_input_field(self) -> str:
        return self.input_correlation_field or self.correlation_field

    @property
    def effective_output_field(self) -> str:
        return self.output_correlation_field or self.correlation_field


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    pairs_queue: TransportResourceConfig = Field(default_factory=TransportResourceConfig)
    timeout_is_anomaly: bool = True
    phase: str = "TRAINING"
    cortex: list[CortexRef] = Field(default_factory=list)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    test_buffer_size: int | None = None  # overrides default test_buffer_size for synthesised IF detector
    detectors: list[DetectorConfig] = Field(default_factory=list)


class AdaptationConfig(BaseModel):
    """Parameters for cortex adaptation strategies. Unused fields per mode are ignored."""
    ema_alpha: float = 0.01
    fine_tuning_interval: int = 100
    fine_tuning_epochs: int = 8
    fine_tuning_lr: float = 0.0001
    replay_ratio: float = 0.3
    replay_buffer_size: int = 500
    sliding_window_size: int = 1000
    sliding_window_interval: int = 200


class CortexConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    grpc_port: int = 50051
    inputs: list[str] = Field(default_factory=list)
    silence_threshold_s: float = 60.0
    parent_cortex: list[CortexRef] = Field(default_factory=list)
    grouping_ttl_s: int = 0
    adaptation_mode: Literal["none", "ema", "fine_tuning", "sliding_window"] = "fine_tuning"
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    test_buffer_size: int | None = None  # overrides model.autoencoder.test_buffer_size
    training_sample_interval_s: float = 5.0


class SentinelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    storage_path: str = "~/.sentinel"
    redis: RedisConfig = Field(default_factory=RedisConfig)
    aws: AwsConfig = Field(default_factory=AwsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    control_plane: ControlPlaneConfig = Field(default_factory=ControlPlaneConfig)
    alarm_queue: TransportResourceConfig = Field(default_factory=TransportResourceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    correlation_engines: list[CorrelationEngineConfig] = Field(default_factory=list)
    agents: list[AgentConfig] = Field(default_factory=list)
    cortex: list[CortexConfig] = Field(default_factory=list)
