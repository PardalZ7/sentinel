"""Sentinel — anomaly detection library for event-driven architectures."""

__version__ = "0.1.0"

from sentinel.config.schema import SentinelConfig
from sentinel.domain.models import (
    Alert,
    AlertSeverity,
    AlertType,
    AnomalyType,
    HeartbeatEvent,
    ModelPhase,
    ProcessedEvent,
    RawMessage,
)

__all__ = [
    "__version__",
    "Alert",
    "AlertSeverity",
    "AlertType",
    "AnomalyType",
    "HeartbeatEvent",
    "ModelPhase",
    "ProcessedEvent",
    "RawMessage",
    "SentinelConfig",
]
