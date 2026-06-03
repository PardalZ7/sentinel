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
  1. Storing incoming input captures keyed by correlation ID (store_input_capture)
  2. Resolving an output message against all stored inputs, producing a CorrelatedPair
     (resolve_pair / resolve_pairs for grouping mode)

All N inputs for a given correlationId must arrive before a pair is emitted.
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
) -> str | None:
    """Store one input stream's capture for a correlationId. Returns the ID or None.

    Uses record_group_source so concurrent arrivals from different input streams
    are merged atomically. The output loop will only resolve once all N inputs
    are present in the group.
    """
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
        return None

    key = make_correlation_key(config.name, correlation_id)
    data = {
        "body": message.body,
        "received_at": message.received_at.isoformat(),
        "size_bytes": message.size_bytes,
    }
    expected_names = [inp.name for inp in config.inputs]
    await store.record_group_source(key, input_cfg.name, data, expected_names, config.correlation_ttl_s)
    return correlation_id


async def store_input_captures_batch(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    input_cfg: InputConfig,
    message: RawMessage,
) -> list[str]:
    """Grouping-mode: store multiple correlationIds from one batch message.

    Each ID in the batch gets its own group entry for this input stream.
    Returns the list of stored IDs, or [] if none were found.
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
    for correlation_id in correlation_ids:
        key = make_correlation_key(config.name, correlation_id)
        await store.record_group_source(key, input_cfg.name, data, expected_names, config.correlation_ttl_s)

    return correlation_ids


async def resolve_pair(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    message: RawMessage,
) -> CorrelatedPair | None:
    """Match output with all stored inputs. Returns CorrelatedPair or None on timeout.

    Handles normal and splitting modes. For grouping mode use resolve_pairs().
    Key is deleted after match in normal mode; kept alive via TTL in splitting mode.
    """
    correlation_id = extract_correlation_id(message.body, config.effective_output_field)
    if not correlation_id:
        logger.warning(
            "missing_correlation_id_on_output",
            engine=config.name,
            field=config.effective_output_field,
        )
        return None

    return await _resolve_single(
        store=store,
        config=config,
        correlation_id=correlation_id,
        message=message,
        delete_key=(config.correlation_mode == "normal"),
    )


async def resolve_pairs(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    message: RawMessage,
) -> list[CorrelatedPair]:
    """Grouping mode: one output message resolves multiple stored input groups.

    The output body must contain a list of correlation IDs at config.correlation_field.
    Each ID is resolved independently and the key is deleted after match.
    """
    ids = extract_correlation_ids(message.body, config.effective_output_field)
    if not ids:
        logger.warning(
            "missing_correlation_ids_on_output",
            engine=config.name,
            field=config.effective_output_field,
        )
        return []

    pairs: list[CorrelatedPair] = []
    for cid in ids:
        pair = await _resolve_single(
            store=store,
            config=config,
            correlation_id=cid,
            message=message,
            delete_key=True,
        )
        if pair is not None:
            pairs.append(pair)
    return pairs


async def _resolve_single(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    correlation_id: str,
    message: RawMessage,
    delete_key: bool,
) -> CorrelatedPair | None:
    key = make_correlation_key(config.name, correlation_id)
    group = await store.get(key)

    if group is None:
        logger.warning("correlation_timeout", engine=config.name, key=key)
        return CorrelatedPair(
            engine_name=config.name,
            correlation_id=correlation_id,
            inputs=[],
            output_body=message.body,
            output_received_at=message.received_at,
            output_size_bytes=message.size_bytes,
            processing_latency_ms=0.0,
            timed_out=True,
        )

    # Build InputCapture list; if any expected input is missing the pair is not ready yet.
    captures: list[InputCapture] = []
    for inp_cfg in config.inputs:
        inp_data = group.get(inp_cfg.name)
        if inp_data is None:
            logger.warning(
                "correlation_incomplete",
                engine=config.name,
                key=key,
                missing_input=inp_cfg.name,
            )
            return CorrelatedPair(
                engine_name=config.name,
                correlation_id=correlation_id,
                inputs=[],
                output_body=message.body,
                output_received_at=message.received_at,
                output_size_bytes=message.size_bytes,
                processing_latency_ms=0.0,
                timed_out=True,
            )
        captures.append(InputCapture(
            name=inp_cfg.name,
            body=inp_data.get("body", {}),
            received_at=datetime.fromisoformat(inp_data["received_at"]),
            size_bytes=inp_data.get("size_bytes", 0),
        ))

    first_received = min(c.received_at for c in captures)
    latency_ms = (message.received_at - first_received).total_seconds() * 1000.0

    if delete_key:
        await store.delete(key)

    return CorrelatedPair(
        engine_name=config.name,
        correlation_id=correlation_id,
        inputs=captures,
        output_body=message.body,
        output_received_at=message.received_at,
        output_size_bytes=message.size_bytes,
        processing_latency_ms=latency_ms,
        timed_out=False,
    )
