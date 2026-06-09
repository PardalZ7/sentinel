from dataclasses import dataclass, field
from datetime import datetime, timezone

from sentinel.domain.models import ModelPhase, ProcessedEvent, WindowSnapshot


@dataclass
class WindowAccumulator:
    layer_name: str
    duration_seconds: int
    window_start: datetime
    events: list[ProcessedEvent] = field(default_factory=list)
    active_agents: set[str] = field(default_factory=set)


def add_event(
    acc: WindowAccumulator,
    event: ProcessedEvent,
    agent_phase: ModelPhase,
) -> WindowAccumulator:
    """Add event to the current window. Discards events from agents in TRAINING."""
    if agent_phase == ModelPhase.TRAINING:
        return acc
    acc.events.append(event)
    acc.active_agents.add(event.agent_name)
    return acc


def should_close(acc: WindowAccumulator, now: datetime) -> bool:
    elapsed = (now - acc.window_start).total_seconds()
    return elapsed >= acc.duration_seconds


def close_window(
    acc: WindowAccumulator,
    bucket_index: int,
) -> tuple[WindowSnapshot, WindowAccumulator]:
    """Close the current window and return a snapshot plus a fresh accumulator."""
    now = datetime.now(tz=timezone.utc)

    total = len(acc.events)
    anomaly_count = sum(1 for e in acc.events if e.is_anomaly)
    anomaly_rate = anomaly_count / total if total > 0 else 0.0
    avg_score = (
        sum(e.anomaly_score for e in acc.events) / total if total > 0 else 0.0
    )
    avg_latency = (
        sum(e.processing_latency_ms for e in acc.events) / total if total > 0 else 0.0
    )
    elapsed_s = (now - acc.window_start).total_seconds()
    events_per_second = total / elapsed_s if elapsed_s > 0 else 0.0

    snapshot = WindowSnapshot(
        layer_name=acc.layer_name,
        window_start=acc.window_start,
        window_end=now,
        bucket_index=bucket_index,
        total_events=total,
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_rate,
        avg_score=avg_score,
        avg_latency_ms=avg_latency,
        events_per_second=events_per_second,
        contributing_agents=sorted(acc.active_agents),
    )

    fresh = WindowAccumulator(
        layer_name=acc.layer_name,
        duration_seconds=acc.duration_seconds,
        window_start=now,
    )

    return snapshot, fresh
