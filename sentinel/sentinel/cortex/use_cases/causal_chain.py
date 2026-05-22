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
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sentinel.domain.models import ProcessedEvent


@dataclass
class CausalDiagnosis:
    root_agent: str | None
    affected_agents: list[str]
    confidence: float
    sequence: list[str]  # agent names in anomaly order


def add_event(window: deque, event: ProcessedEvent) -> deque:
    """Add a ProcessedEvent to the sliding window (anomalies only tracked)."""
    if event.is_anomaly:
        window.append(event)
    return window


def analyze_causality(window: deque, window_s: float = 30.0) -> CausalDiagnosis:
    """Analyze a window of anomaly events to identify root cause and cascade.

    Simple algorithm:
    - Filter events within window_s of the most recent event
    - Order by timestamp (ascending)
    - First event is the root cause candidate
    - Confidence proportional to number of affected agents
    """
    if not window:
        return CausalDiagnosis(
            root_agent=None, affected_agents=[], confidence=0.0, sequence=[]
        )

    events = list(window)
    if not events:
        return CausalDiagnosis(
            root_agent=None, affected_agents=[], confidence=0.0, sequence=[]
        )

    # Sort by output_sent_at ascending
    events_sorted = sorted(events, key=lambda e: e.output_sent_at)
    latest_time = events_sorted[-1].output_sent_at

    # Filter to events within window_s of the latest
    relevant = [
        e
        for e in events_sorted
        if (latest_time - e.output_sent_at).total_seconds() <= window_s
    ]

    if not relevant:
        return CausalDiagnosis(
            root_agent=None, affected_agents=[], confidence=0.0, sequence=[]
        )

    # Build ordered sequence of unique agents
    seen: set[str] = set()
    sequence: list[str] = []
    for e in relevant:
        if e.agent_name not in seen:
            seen.add(e.agent_name)
            sequence.append(e.agent_name)

    root_agent = sequence[0] if sequence else None
    affected_agents = sequence[1:] if len(sequence) > 1 else []

    # Confidence: higher when more agents are affected in a tight chain
    confidence = min(1.0, len(sequence) / max(1, len(set(e.agent_name for e in events))))

    return CausalDiagnosis(
        root_agent=root_agent,
        affected_agents=affected_agents,
        confidence=confidence,
        sequence=sequence,
    )
