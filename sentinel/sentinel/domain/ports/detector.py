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

"""Strategy interface for anomaly detection algorithms.

All anomaly detection algorithms used by agents must implement IDetector.
Implementations are intentionally stateless — mutable runtime state lives in
DetectorState, which is owned and mutated by the agent use-case layer.

This separation allows:
- The same detector object to evaluate challenger models against the current
  champion without state confusion.
- Clean champion/challenger selection: the detector scores both models against
  the same test buffer and returns a pure result.

Score semantics differ by algorithm family:
  IsolationForest:  lower (more negative) score = more anomalous  → is_anomaly: score < threshold
  cVAE / MAF / NRI: higher score = more anomalous (reconstruction error / neg log-likelihood)
                                                                  → is_anomaly: score > threshold

Each implementation's is_anomaly() encapsulates the correct direction so callers
never need to know which family a detector belongs to.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sentinel.domain.models import ModelPhase


@dataclass
class DetectorState:
    """Mutable runtime state for a single detector within an agent.

    One DetectorState per detector per agent. Owned exclusively by the agent
    use-case layer; never mutated directly from outside.

    Fields are analogous to the old AgentState fields, now scoped per-detector
    so each algorithm trains and transitions phases independently.
    """

    name: str
    algorithm: str
    phase: ModelPhase
    model: Any                                              # None until first training cycle
    training_buffer: list = field(default_factory=list)
    test_buffer: deque = field(default_factory=deque)
    test_sample_rate: float = 0.05
    last_test_result: dict = field(default_factory=dict)
    champion_fp_rate: float = 1.0                          # 1.0 = no champion yet
    challengers_rejected: int = 0
    max_challengers_rejected: int = 0                      # 0 = unlimited; >0 = stop training after N consecutive rejections
    training_count: int = 0                                # incremented each time a challenger is trained (TRAINING phase only)
    auto_infer_fp_threshold: float | None = None           # None = disabled; float = auto-transition to INFERENCE when champion_fp_rate < threshold
    training_in_progress: bool = False                     # True while background training task is running
    gating_detector: str | None = None                     # name of detector used to filter contaminated training samples

    @property
    def is_degraded(self) -> bool:
        """True when max_challengers_rejected is set and consecutive rejections reached it."""
        return self.max_challengers_rejected > 0 and self.challengers_rejected >= self.max_challengers_rejected


class IDetector(ABC):
    """Strategy interface for anomaly detection algorithms.

    Implementations are stateless configuration objects. All mutable runtime
    state is stored in DetectorState and passed in by the caller.

    Conventions for implementors:
    - extract_features() must handle missing payload keys gracefully (return 0.0)
    - fit() must not mutate self
    - score() must not mutate self
    - is_anomaly() must be consistent with the algorithm's score direction

    The event_dict passed to extract_features() always contains:
      processing_latency_ms  — float, milliseconds
      input_size_bytes       — int
      output_size_bytes      — int
      payload_schema_hash    — str, 16-char hex

    Semantic detectors (cVAE, MAF, NRI) may also use:
      _input_body            — dict, raw input message body (may be absent)
      _output_body           — dict, raw output message body (may be absent)

    The sentinel core always populates _input_body and _output_body.
    Detectors that do not need them simply ignore those keys.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this detector within its agent."""

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Algorithm identifier (e.g. 'isolation_forest', 'cvae', 'maf', 'nri')."""

    @property
    @abstractmethod
    def training_window(self) -> int:
        """Minimum sample count before the first training cycle is triggered."""

    @property
    @abstractmethod
    def test_buffer_size(self) -> int:
        """Maximum number of hold-out samples kept for champion evaluation."""

    @abstractmethod
    def extract_features(self, event_dict: dict) -> list[float]:
        """Return a numeric feature vector for a processed event dict.

        Must be deterministic and side-effect-free.
        """

    @abstractmethod
    def fit(self, samples: list[dict]) -> Any:
        """Train a new model on the provided samples.

        Returns the trained model object. Does NOT mutate self.
        The caller is responsible for storing the result in DetectorState.
        """

    @abstractmethod
    def score(self, model: Any, event_dict: dict) -> float:
        """Compute the anomaly score for a single event.

        Score semantics are algorithm-specific. Use is_anomaly() to interpret.
        """

    @abstractmethod
    def is_anomaly(self, score: float, model: Any = None) -> bool:
        """Return True if the score indicates an anomaly.

        Encapsulates algorithm-specific threshold direction (< vs >).
        The optional model parameter allows detectors to use a model-embedded
        auto-calibrated threshold (e.g. cVAE stores auto_threshold in the model dict).
        """
