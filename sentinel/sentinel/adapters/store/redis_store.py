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

import json
import time

from sentinel.domain.errors import RedisUnavailableError
from sentinel.domain.ports.correlation_store import ICorrelationStore
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

# Atomically merges one source's snapshot into a group stored as JSON.
# KEYS[1]  = Redis key
# ARGV[1]  = source_name
# ARGV[2]  = JSON-encoded event snapshot
# ARGV[3]  = JSON-encoded list of expected source names (used only on first write)
# ARGV[4]  = TTL in seconds
# ARGV[5]  = current time in milliseconds (for _created_at_ms on new groups)
# Returns the JSON-encoded updated group.
_RECORD_GROUP_SOURCE_LUA = """
local existing = redis.call('GET', KEYS[1])
local group
if existing == false then
    local sources = cjson.decode(ARGV[3])
    group = {}
    for _, s in ipairs(sources) do
        group[s] = cjson.null
    end
    group['_created_at_ms'] = tonumber(ARGV[5])
else
    group = cjson.decode(existing)
end
group[ARGV[1]] = cjson.decode(ARGV[2])
local encoded = cjson.encode(group)
redis.call('SETEX', KEYS[1], tonumber(ARGV[4]), encoded)
return encoded
"""

# Atomically claims a complete group: checks all expected sources are present
# (non-null), deletes the key, and returns 1. Returns 0 if incomplete or gone.
# KEYS[1]  = Redis key
# ARGV[1]  = JSON-encoded list of expected source names
_TRY_CLAIM_COMPLETE_GROUP_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing == false then
    return 0
end
local group = cjson.decode(existing)
local sources = cjson.decode(ARGV[1])
for _, s in ipairs(sources) do
    if group[s] == nil or group[s] == cjson.null then
        return 0
    end
end
redis.call('DEL', KEYS[1])
return 1
"""


class RedisCorrelationStore(ICorrelationStore):
    """ICorrelationStore implementation using redis.asyncio."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: str | None = None) -> None:
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._client = None
        self._record_group_script = None
        self._try_claim_script = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.Redis(
                    host=self._host,
                    port=self._port,
                    db=self._db,
                    password=self._password,
                    decode_responses=True,
                )
            except ImportError as e:
                raise RedisUnavailableError(
                    "redis package is required. Install with: pip install redis"
                ) from e
        return self._client

    async def save(self, key: str, data: dict, ttl_s: int) -> None:
        """Persist data as JSON with the given TTL."""
        client = self._get_client()
        try:
            await client.setex(key, ttl_s, json.dumps(data))
        except Exception as exc:
            raise RedisUnavailableError(f"Redis save failed: {exc}") from exc

    async def get(self, key: str) -> dict | None:
        """Retrieve and deserialize a value by key. Returns None if not found."""
        client = self._get_client()
        try:
            value = await client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as exc:
            raise RedisUnavailableError(f"Redis get failed: {exc}") from exc

    async def delete(self, key: str) -> None:
        """Delete a key from Redis."""
        client = self._get_client()
        try:
            await client.delete(key)
        except Exception as exc:
            raise RedisUnavailableError(f"Redis delete failed: {exc}") from exc

    async def record_group_source(
        self,
        key: str,
        source_name: str,
        event_snapshot: dict,
        expected_sources: list[str],
        ttl_s: int,
    ) -> dict:
        """Atomically add one source's snapshot to a group via a Lua script."""
        client = self._get_client()
        try:
            if self._record_group_script is None:
                self._record_group_script = client.register_script(_RECORD_GROUP_SOURCE_LUA)
            now_ms = int(time.time() * 1000)
            result = await self._record_group_script(
                keys=[key],
                args=[
                    source_name,
                    json.dumps(event_snapshot),
                    json.dumps(expected_sources),
                    str(ttl_s),
                    str(now_ms),
                ],
            )
            return json.loads(result)
        except Exception as exc:
            raise RedisUnavailableError(f"Redis record_group_source failed: {exc}") from exc

    async def try_claim_complete_group(
        self,
        key: str,
        expected_sources: list[str],
    ) -> bool:
        """Atomically claim a complete group via a Lua script. Returns True if claimed."""
        client = self._get_client()
        try:
            if self._try_claim_script is None:
                self._try_claim_script = client.register_script(_TRY_CLAIM_COMPLETE_GROUP_LUA)
            result = await self._try_claim_script(
                keys=[key],
                args=[json.dumps(expected_sources)],
            )
            return bool(result)
        except Exception as exc:
            raise RedisUnavailableError(f"Redis try_claim_complete_group failed: {exc}") from exc

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
