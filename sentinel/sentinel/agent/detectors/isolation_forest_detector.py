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

"""IsolationForest-based anomaly detector.

Detects operational anomalies: abnormal latency, unusual message sizes, and
unexpected schema changes.  Does not inspect payload field values.

This detector wraps the existing sentinel.agent.isolation_forest module and
adapts it to the IDetector strategy interface.
"""

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector


@lru_cache(maxsize=256)
def _hash_schema(schema_hash: str) -> float:
    """Convert a hex schema hash string to a bounded integer feature.

    Cached because the schema is stable for a given payload shape — the same
    string will appear thousands of times per session.
    """
    return float(int(hashlib.md5(schema_hash.encode()).hexdigest(), 16) % int(1e6))


class IsolationForestDetector(IDetector):
    """Anomaly detector backed by scikit-learn's IsolationForest.

    Features used:
      - processing_latency_ms
      - input_size_bytes
      - output_size_bytes
      - payload_schema_hash (hashed to integer)

    Score semantics: decision_function returns values in [-1, 0].
    More negative = more anomalous.  Anomaly when score < threshold (e.g. -0.1).
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def algorithm(self) -> str:
        return "isolation_forest"

    @property
    def training_window(self) -> int:
        return self._config.training_window

    @property
    def test_buffer_size(self) -> int:
        return self._config.test_buffer_size

    def extract_features(self, event_dict: dict) -> list[float]:
        latency_ms = float(event_dict.get("processing_latency_ms", 0.0))
        input_size = float(event_dict.get("input_size_bytes", 0))
        output_size = float(event_dict.get("output_size_bytes", 0))
        schema_hash = event_dict.get("payload_schema_hash", "")
        hash_int = _hash_schema(schema_hash) if schema_hash else 0.0
        return [latency_ms, input_size, output_size, hash_int]

    def fit(self, samples: list[dict]) -> IsolationForest:
        contamination = self._config.contamination
        if contamination != "auto":
            contamination = float(contamination)

        model = IsolationForest(
            n_estimators=self._config.n_estimators,
            contamination=contamination,
            random_state=42,
        )
        features = [self.extract_features(s) for s in samples]
        X = np.array(features)
        model.fit(X)
        return model

    def score(self, model: Any, event_dict: dict) -> float:
        features = self.extract_features(event_dict)
        X = np.array([features])
        scores = model.decision_function(X)
        return float(scores[0])

    def batch_score(self, model: Any, samples: list[dict]) -> list[float]:
        """Score all samples in a single vectorized call.

        Used by champion evaluation to avoid N individual decision_function calls.
        """
        features = [self.extract_features(s) for s in samples]
        X = np.array(features)
        return [float(s) for s in model.decision_function(X)]

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        return score < self._config.anomaly_threshold
