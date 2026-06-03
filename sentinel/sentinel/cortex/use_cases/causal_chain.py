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

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from sentinel.domain.models import TemporalAnomalySignal


@dataclass
class CausalDiagnosis:
    root_layer: str | None
    affected_layers: list[str]
    confidence: float
    sequence: list[str]


@dataclass
class _CausalEntry:
    source_name: str
    timestamp: datetime
    score: float


def add_temporal_signal(
    window: deque,
    signal: TemporalAnomalySignal,
) -> deque:
    """Add a TemporalAnomalySignal to the causal window.

    Uses window_end as the event timestamp and the highest z-score as the score.
    """
    score = max(
        abs(signal.anomaly_rate_z),
        abs(signal.volume_z),
        abs(signal.avg_score_z),
        abs(signal.avg_latency_z),
    )
    window.append(_CausalEntry(
        source_name=signal.layer_name,
        timestamp=signal.window_end,
        score=score,
    ))
    return window


def analyze_causality(window: deque, window_s: float = 30.0) -> CausalDiagnosis:
    """Analyze the causal window to identify root layer and cascade sequence.

    - Filters entries within window_s of the most recent entry
    - Orders by timestamp ascending — first is root cause candidate
    - Confidence proportional to number of distinct layers involved
    """
    if not window:
        return CausalDiagnosis(root_layer=None, affected_layers=[], confidence=0.0, sequence=[])

    entries: list[_CausalEntry] = list(window)
    entries_sorted = sorted(entries, key=lambda e: e.timestamp)
    latest_time = entries_sorted[-1].timestamp

    relevant = [
        e for e in entries_sorted
        if (latest_time - e.timestamp).total_seconds() <= window_s
    ]

    if not relevant:
        return CausalDiagnosis(root_layer=None, affected_layers=[], confidence=0.0, sequence=[])

    seen: set[str] = set()
    sequence: list[str] = []
    for e in relevant:
        if e.source_name not in seen:
            seen.add(e.source_name)
            sequence.append(e.source_name)

    root_layer = sequence[0] if sequence else None
    affected_layers = sequence[1:] if len(sequence) > 1 else []
    total_sources = len({e.source_name for e in entries})
    confidence = min(1.0, len(sequence) / max(1, total_sources))

    return CausalDiagnosis(
        root_layer=root_layer,
        affected_layers=affected_layers,
        confidence=confidence,
        sequence=sequence,
    )
