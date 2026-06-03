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

"""Correlation use case for CorrelationEngine.

Responsible for:
  1. Storing incoming input captures keyed by correlation ID (store_input_capture /
     store_input_captures_batch). Returns the group snapshot from Redis so the caller
     can immediately check completeness without a second round-trip.
  2. Building a CorrelatedPair once ALL expected inputs have arrived (try_build_pair).
     Uses an atomic claim (try_claim_complete_group) so only one concurrent loop ever
     publishes a pair for a given correlationId.

Resolution is triggered from the input loop — there is no separate output loop.
The Agent has no dependency on this module — it only consumes CorrelatedPair.
"""

from datetime import datetime

from sentinel.agent.correlator import (
    extract_correlation_id,
    extract_correlation_ids,
    make_correlation_key,
)
from sentinel.config.schema import CorrelationEngineConfig, InputConfig
from sentinel.domain.models import CorrelatedPair, InputCapture, RawMessage
from sentinel.domain.ports.correlation_store import ICorrelationStore
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


async def store_input_capture(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    input_cfg: InputConfig,
    message: RawMessage,
) -> tuple[str, dict] | tuple[None, None]:
    """Store one input stream's capture. Returns (correlation_id, updated_group) or (None, None)."""
    field = config.effective_input_field(input_cfg)
    correlation_id = extract_correlation_id(message.body, field)
    if not correlation_id:
        logger.warning(
            "missing_correlation_id_on_input",
            engine=config.name,
            input=input_cfg.name,
            field=field,
            body_keys=list(message.body.keys()) if isinstance(message.body, dict) else type(message.body).__name__,
            body_preview=str(message.body)[:200],
        )
        return None, None

    key = make_correlation_key(config.name, correlation_id)
    data = {
        "body": message.body,
        "received_at": message.received_at.isoformat(),
        "size_bytes": message.size_bytes,
    }
    expected_names = [inp.name for inp in config.inputs]
    group = await store.record_group_source(key, input_cfg.name, data, expected_names, config.correlation_ttl_s)
    return correlation_id, group


async def store_input_captures_batch(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    input_cfg: InputConfig,
    message: RawMessage,
) -> list[tuple[str, dict]]:
    """Grouping-mode: store multiple correlationIds from one batch message.

    Returns list of (correlation_id, updated_group) pairs, one per ID stored.
    """
    field = config.effective_input_field(input_cfg)
    correlation_ids = extract_correlation_ids(message.body, field)
    if not correlation_ids:
        logger.warning(
            "missing_correlation_ids_on_input",
            engine=config.name,
            input=input_cfg.name,
            field=field,
            body_keys=list(message.body.keys()) if isinstance(message.body, dict) else type(message.body).__name__,
            body_preview=str(message.body)[:200],
        )
        return []

    data = {
        "body": message.body,
        "received_at": message.received_at.isoformat(),
        "size_bytes": message.size_bytes,
    }
    expected_names = [inp.name for inp in config.inputs]
    results: list[tuple[str, dict]] = []
    for correlation_id in correlation_ids:
        key = make_correlation_key(config.name, correlation_id)
        group = await store.record_group_source(key, input_cfg.name, data, expected_names, config.correlation_ttl_s)
        results.append((correlation_id, group))

    return results


def _is_group_complete(group: dict, config: CorrelationEngineConfig) -> bool:
    """Return True if all expected inputs are present (non-null) in the group."""
    for inp_cfg in config.inputs:
        if not group.get(inp_cfg.name):
            return False
    return True


async def try_build_pair(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    correlation_id: str,
    group: dict,
) -> CorrelatedPair | None:
    """Attempt to build a CorrelatedPair from the given group snapshot.

    First checks completeness in-memory (no Redis round-trip). If complete,
    performs an atomic claim (try_claim_complete_group) to ensure only one
    concurrent caller builds and publishes the pair. Returns None if the group
    is incomplete or another caller already claimed it.
    """
    if not _is_group_complete(group, config):
        return None

    key = make_correlation_key(config.name, correlation_id)
    expected_names = [inp.name for inp in config.inputs]
    claimed = await store.try_claim_complete_group(key, expected_names)
    if not claimed:
        return None

    captures: list[InputCapture] = []
    for inp_cfg in config.inputs:
        inp_data = group[inp_cfg.name]
        captures.append(InputCapture(
            name=inp_cfg.name,
            body=inp_data.get("body", {}),
            received_at=datetime.fromisoformat(inp_data["received_at"]),
            size_bytes=inp_data.get("size_bytes", 0),
        ))

    first_received = min(c.received_at for c in captures)
    last_received = max(c.received_at for c in captures)
    latency_ms = (last_received - first_received).total_seconds() * 1000.0

    return CorrelatedPair(
        engine_name=config.name,
        correlation_id=correlation_id,
        inputs=captures,
        output_body={},
        output_received_at=last_received,
        output_size_bytes=0,
        processing_latency_ms=latency_ms,
        timed_out=False,
    )
