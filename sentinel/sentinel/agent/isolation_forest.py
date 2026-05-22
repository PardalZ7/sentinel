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

import hashlib

import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel.config.schema import ModelIsolationForestConfig


def create_model(config: ModelIsolationForestConfig) -> IsolationForest:
    """Instantiate a new IsolationForest with the given config."""
    contamination = config.contamination
    if contamination != "auto":
        contamination = float(contamination)

    return IsolationForest(
        n_estimators=config.n_estimators,
        contamination=contamination,
        random_state=42,
    )


def extract_features(event: dict) -> list[float]:
    """Extract numeric features from a processed event dict.

    Features: latency_ms, input_size, output_size, schema_hash_int (hash % 1e6)
    """
    latency_ms = float(event.get("processing_latency_ms", 0.0))
    input_size = float(event.get("input_size_bytes", 0))
    output_size = float(event.get("output_size_bytes", 0))

    schema_hash = event.get("payload_schema_hash", "")
    if schema_hash:
        hash_int = int(hashlib.md5(schema_hash.encode()).hexdigest(), 16) % int(1e6)
    else:
        hash_int = 0.0

    return [latency_ms, input_size, output_size, float(hash_int)]


def fit_model(model: IsolationForest, samples: list[dict]) -> IsolationForest:
    """Fit the model on a list of processed event dicts."""
    features = [extract_features(s) for s in samples]
    X = np.array(features)
    model.fit(X)
    return model


def score_event(model: IsolationForest, event: dict) -> float:
    """Return the anomaly score for a single event dict.

    IsolationForest.score_samples returns negative values; more negative = more anomalous.
    """
    features = extract_features(event)
    X = np.array([features])
    scores = model.decision_function(X)
    return float(scores[0])


def is_anomaly(score: float, threshold: float) -> bool:
    """Return True if the score is below the threshold (anomalous)."""
    return score < threshold
