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

"""Cortex grouping buffer: accumulates per-source events for a correlation_id.

Design:
- Stored in Redis under key  cortex:{cortex_name}:group:{correlation_id}
- Value: JSON dict  { "<source>": {event_snapshot} | null, "_created_at_ms": int }
- TTL is set on first write; Redis handles expiry silently.
- When all declared sources have reported, the group is "complete".
- Splitting-mode sources: the first event from a source marks it present;
  subsequent events for the same (cid, source) update the stored snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.domain.ports.correlation_store import ICorrelationStore


def _key(cortex_name: str, correlation_id: str) -> str:
    return f"cortex:{cortex_name}:group:{correlation_id}"


def _event_snapshot(event: object) -> dict:
    """Extract the fields relevant for grouping from a ProcessedEvent."""
    return {
        "agent_name": event.agent_name,
        "correlation_id": event.correlation_id,
        "input_received_at": event.input_received_at.isoformat(),
        "output_sent_at": event.output_sent_at.isoformat(),
        "processing_latency_ms": round(event.processing_latency_ms, 2),
        "anomaly_score": round(event.anomaly_score, 4),
        "is_anomaly": event.is_anomaly,
        "anomaly_type": event.anomaly_type.value if event.anomaly_type else "",
        "payload_schema_hash": event.payload_schema_hash,
        "input_size_bytes": event.input_size_bytes,
        "output_size_bytes": event.output_size_bytes,
    }


async def record_source_event(
    store: "ICorrelationStore",
    cortex_name: str,
    correlation_id: str,
    source_name: str,
    event: object,
    expected_sources: list[str],
    ttl_s: int,
) -> dict:
    """Record one source's event in the group; return the updated group dict."""
    key = _key(cortex_name, correlation_id)
    return await store.record_group_source(
        key=key,
        source_name=source_name,
        event_snapshot=_event_snapshot(event),
        expected_sources=expected_sources,
        ttl_s=ttl_s,
    )


def is_complete(group: dict, expected_sources: list[str]) -> bool:
    """Return True when every expected source has a non-None entry."""
    return all(group.get(s) is not None for s in expected_sources)


async def get_group(
    store: "ICorrelationStore",
    cortex_name: str,
    correlation_id: str,
) -> dict | None:
    return await store.get(_key(cortex_name, correlation_id))


async def delete_group(
    store: "ICorrelationStore",
    cortex_name: str,
    correlation_id: str,
) -> None:
    await store.delete(_key(cortex_name, correlation_id))


def build_grouped_event(
    group: dict,
    cortex_name: str,
    correlation_id: str,
    expected_sources: list[str],
) -> object:
    """Build a ProcessedEvent representing the aggregated group.

    - is_anomaly  = any source was anomalous
    - anomaly_score = max (most negative) score across sources
    - processing_latency_ms = span from earliest input_received_at to latest output_sent_at
    - input_body = the group snapshot (used for dashboard detail view)
    """
    from datetime import datetime, timezone

    from sentinel.domain.models import AnomalyType, ProcessedEvent

    source_events = [group[s] for s in expected_sources if group.get(s) is not None]

    any_anomaly = any(e["is_anomaly"] for e in source_events)
    max_score = min((e["anomaly_score"] for e in source_events), default=0.0)

    # Span latency: earliest input → latest output
    input_ts_list = []
    output_ts_list = []
    for e in source_events:
        try:
            input_ts_list.append(datetime.fromisoformat(e["input_received_at"]))
            output_ts_list.append(datetime.fromisoformat(e["output_sent_at"]))
        except Exception:
            pass

    if input_ts_list and output_ts_list:
        earliest_input = min(input_ts_list)
        latest_output = max(output_ts_list)
        latency_ms = (latest_output - earliest_input).total_seconds() * 1000.0
    else:
        earliest_input = datetime.now(tz=timezone.utc)
        latest_output = earliest_input
        latency_ms = 0.0

    # Include source events that were anomalous in the input_body snapshot
    anomalous_sources = {s: group[s] for s in expected_sources if group.get(s) and group[s]["is_anomaly"]}
    all_sources = {s: group[s] for s in expected_sources if group.get(s)}

    return ProcessedEvent(
        agent_name=cortex_name,
        correlation_id=correlation_id,
        input_received_at=earliest_input,
        output_sent_at=latest_output,
        processing_latency_ms=round(latency_ms, 2),
        anomaly_score=round(max_score, 4),
        is_anomaly=any_anomaly,
        anomaly_type=AnomalyType.SCORE if any_anomaly else None,
        payload_schema_hash="",
        input_size_bytes=sum(e.get("input_size_bytes", 0) for e in source_events),
        output_size_bytes=sum(e.get("output_size_bytes", 0) for e in source_events),
        input_body=all_sources if any_anomaly else None,
        output_body={"anomalous_sources": anomalous_sources} if any_anomaly else None,
    )
