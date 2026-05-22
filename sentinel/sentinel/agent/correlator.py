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

from collections.abc import Awaitable
from functools import lru_cache
from typing import Any

from sentinel.domain.ports.correlation_store import ICorrelationStore


def make_correlation_key(agent_name: str, correlation_id: str) -> str:
    """Build a Redis key from agent name and correlation ID."""
    return f"{agent_name}:{correlation_id}"


@lru_cache(maxsize=256)
def _split_path(field_path: str) -> tuple[str, ...]:
    """Split a dot-notation field path into parts.

    Cached because each agent uses a fixed field_path for every message —
    splitting the same string thousands of times is pure waste.
    """
    return tuple(field_path.split("."))


def extract_correlation_id(body: dict, field_path: str) -> str | None:
    """Extract a single correlation ID from a nested dict using dot-notation field path.

    Example: extract_correlation_id(body, "meta.correlationId")
    Returns None if the field is missing or the path traverses a list.
    """
    parts = _split_path(field_path)
    current: Any = body
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    if isinstance(current, list):
        return None
    return str(current) if current is not None else None


def extract_correlation_ids(body: dict, field_path: str) -> list[str]:
    """Extract one or more correlation IDs from a message body.

    Handles both simple scalar paths and array traversal:
    - "correlationId"          → ["abc-123"]
    - "records.correlationId"  → ["id1", "id2", ...] when records is a list

    When a path segment resolves to a list, the remaining path is extracted
    from each element in that list.
    """
    parts = _split_path(field_path)
    current: Any = body

    for i, part in enumerate(parts):
        if not isinstance(current, dict):
            return []
        current = current.get(part)
        if current is None:
            return []
        if isinstance(current, list):
            remaining = parts[i + 1:]
            if not remaining:
                return [str(v) for v in current if v is not None]
            results = []
            for item in current:
                val: Any = item
                for rp in remaining:
                    if not isinstance(val, dict):
                        val = None
                        break
                    val = val.get(rp)
                if val is not None:
                    results.append(str(val))
            return results

    if isinstance(current, list):
        return [str(v) for v in current if v is not None]
    if current is not None:
        return [str(current)]
    return []


async def store_input(
    store: ICorrelationStore, key: str, data: dict, ttl_s: int
) -> None:
    """Persist input message data in the correlation store."""
    await store.save(key, data, ttl_s)


async def retrieve_input(
    store: ICorrelationStore, key: str
) -> dict | None:
    """Retrieve a previously stored input message from the correlation store."""
    return await store.get(key)
