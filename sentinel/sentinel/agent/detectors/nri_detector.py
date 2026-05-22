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

"""Neural Relational Inference (NRI) anomaly detector — stable interface stub.

NRI learns an explicit dependency graph between fields, providing the most
interpretable anomaly signal: "field X depends on field Y, but that relationship
has broken."

This module exposes the full IDetector interface with a stable contract so
callers (launcher, process_message, tests) can reference and configure this
detector today.  The training and scoring logic will be filled in a future
iteration without any changes to the surrounding architecture.

Current behaviour:
  - fit()   raises NotImplementedError (logged as warning, not crash)
  - score() returns 0.0 (safe no-op — no false positives)
  - is_anomaly() always returns False

To avoid runtime crashes when NRI is misconfigured, build_detector() in factory.py
will log a warning if NRI is used without a field_map.
"""

from typing import Any

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class NRIDetector(IDetector):
    """Neural Relational Inference detector (stub implementation).

    Interface is stable and safe to use in configuration.
    Training raises NotImplementedError; scoring is a safe no-op.

    Recommended sentinel.json entry (for future use):
      {
        "name": "nri_relations",
        "algorithm": "nri",
        "phase": "TRAINING",
        "training_window": 3000,
        "test_buffer_size": 300,
        "anomaly_threshold": 0.1,
        "field_map": {
          "input": ["field_a", "field_b"],
          "output": ["field_c", "field_d"]
        }
      }
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        if config.field_map is None:
            logger.warning(
                "nri_no_field_map",
                detector=config.name,
                msg="NRI detector works best with an explicit field_map in sentinel.json.",
            )

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def algorithm(self) -> str:
        return "nri"

    @property
    def training_window(self) -> int:
        return self._config.training_window

    @property
    def test_buffer_size(self) -> int:
        return self._config.test_buffer_size

    def extract_features(self, event_dict: dict) -> list[float]:
        if self._config.field_map is None:
            return []
        fm = self._config.field_map
        all_fields = fm.input + fm.output
        merged: dict = {}
        for key in ("_input_body", "_output_body"):
            merged.update(event_dict.get(key) or {})
        return [float(merged.get(f, 0.0)) for f in all_fields]

    def fit(self, samples: list[dict]) -> Any:
        raise NotImplementedError(
            f"NRI detector '{self.name}' training is not yet implemented. "
            "The interface is stable — this will be filled in a future release."
        )

    def score(self, model: Any, event_dict: dict) -> float:
        return 0.0

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        return False
