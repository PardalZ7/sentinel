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
  1. Storing incoming input messages keyed by correlation ID (store_request)
  2. Resolving an output message against its stored input, producing a CorrelatedPair
     (resolve_pair / resolve_pairs for grouping mode)

The Agent has no dependency on this module — it only consumes CorrelatedPair.
"""

from datetime import datetime

from sentinel.agent.correlator import (
    extract_correlation_id,
    extract_correlation_ids,
    make_correlation_key,
    retrieve_input,
    store_input,
)
from sentinel.config.schema import CorrelationEngineConfig
from sentinel.domain.models import CorrelatedPair, RawMessage
from sentinel.domain.ports.correlation_store import ICorrelationStore
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


async def store_request(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    message: RawMessage,
) -> str | None:
    """Persist an input message under its correlation ID. Returns the ID or None."""
    correlation_id = extract_correlation_id(message.body, config.correlation_field)
    if not correlation_id:
        logger.warning(
            "missing_correlation_id_on_input",
            engine=config.name,
            field=config.correlation_field,
        )
        return None

    key = make_correlation_key(config.name, correlation_id)
    data = {
        "body": message.body,
        "received_at": message.received_at.isoformat(),
        "size_bytes": message.size_bytes,
    }
    await store_input(store, key, data, config.correlation_ttl_s)
    return correlation_id


async def resolve_pair(
    store: ICorrelationStore,
    config: CorrelationEngineConfig,
    message: RawMessage,
) -> CorrelatedPair | None:
    """Match output with stored input. Returns CorrelatedPair or None on timeout.

    Handles normal and splitting modes. For grouping mode use resolve_pairs().
    Key is deleted after match in normal mode; kept alive via TTL in splitting mode.
    """
    correlation_id = extract_correlation_id(message.body, config.correlation_field)
    if not correlation_id:
        logger.warning(
            "missing_correlation_id_on_output",
            engine=config.name,
            field=config.correlation_field,
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
    """Grouping mode: one output message resolves multiple stored inputs.

    The output body must contain a list of correlation IDs at config.correlation_field.
    Each ID is resolved independently and the key is deleted after match.
    """
    ids = extract_correlation_ids(message.body, config.correlation_field)
    if not ids:
        logger.warning(
            "missing_correlation_ids_on_output",
            engine=config.name,
            field=config.correlation_field,
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
    input_data = await retrieve_input(store, key)

    if input_data is None:
        logger.warning("correlation_timeout", engine=config.name, key=key)
        return CorrelatedPair(
            engine_name=config.name,
            correlation_id=correlation_id,
            input_body={},
            output_body=message.body,
            input_received_at=message.received_at,
            output_received_at=message.received_at,
            input_size_bytes=0,
            output_size_bytes=message.size_bytes,
            processing_latency_ms=0.0,
            timed_out=True,
        )

    input_received_at = datetime.fromisoformat(input_data["received_at"])
    latency_ms = (message.received_at - input_received_at).total_seconds() * 1000.0

    if delete_key:
        await store.delete(key)

    return CorrelatedPair(
        engine_name=config.name,
        correlation_id=correlation_id,
        input_body=input_data.get("body", {}),
        output_body=message.body,
        input_received_at=input_received_at,
        output_received_at=message.received_at,
        input_size_bytes=input_data.get("size_bytes", 0),
        output_size_bytes=message.size_bytes,
        processing_latency_ms=latency_ms,
        timed_out=False,
    )
