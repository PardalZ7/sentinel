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

import logging
import os
from typing import Any

import structlog

# Events suppressed in normal mode — only visible when SENTINEL_VERBOSE=detailed
_VERBOSE_ONLY_EVENTS: frozenset[str] = frozenset(
    [
        "grpc_report_event",
        "grpc_report_heartbeat",
        "grpc_report_temporal_heartbeat",
        "denial_rule_hit",
        "denial_rule_field_missing",
        "denial_rule_eval_error",
        "sqs_messages_received",
        "agent_pairs_batch_received",
    ]
)

_VERBOSE_MODE = os.getenv("SENTINEL_VERBOSE", "").strip().lower() == "detailed"


def _verbose_filter(
    logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    if not _VERBOSE_MODE and event_dict.get("event") in _VERBOSE_ONLY_EVENTS:
        raise structlog.DropEvent()
    return event_dict


def configure_logging() -> None:
    """Configure structlog for the application. Call once at startup."""
    log_format = os.getenv("SENTINEL_LOG_FORMAT", "json")

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _verbose_filter,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if log_format == "pretty":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
