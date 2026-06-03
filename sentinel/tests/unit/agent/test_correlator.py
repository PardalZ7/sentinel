import pytest

from sentinel.agent.correlator import (
    extract_correlation_id,
    make_correlation_key,
    retrieve_input,
    store_input,
)
from sentinel.domain.ports.correlation_store import ICorrelationStore


class MockCorrelationStore(ICorrelationStore):
    def __init__(self):
        self._store: dict[str, dict] = {}

    async def save(self, key: str, data: dict, ttl_s: int) -> None:
        self._store[key] = data

    async def get(self, key: str) -> dict | None:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def record_group_source(self, key, source, data, expected_sources, ttl_s):
        group = self._store.get(key)
        if group is None:
            group = {s: None for s in expected_sources}
        group[source] = data
        self._store[key] = group
        return group

    async def try_claim_complete_group(self, key, expected_sources):
        group = self._store.get(key)
        if group is None:
            return False
        if any(group.get(s) is None for s in expected_sources):
            return False
        self._store.pop(key)
        return True


# ── make_correlation_key ──────────────────────────────────────────────────────

def test_make_correlation_key_basic():
    key = make_correlation_key("agent01", "abc-123")
    assert key == "agent01:abc-123"


def test_make_correlation_key_empty_ids():
    key = make_correlation_key("", "")
    assert key == ":"


# ── extract_correlation_id ────────────────────────────────────────────────────

def test_extract_correlation_id_top_level():
    body = {"correlationId": "xyz-789"}
    result = extract_correlation_id(body, "correlationId")
    assert result == "xyz-789"


def test_extract_correlation_id_dot_notation():
    body = {"meta": {"correlationId": "deep-id"}}
    result = extract_correlation_id(body, "meta.correlationId")
    assert result == "deep-id"


def test_extract_correlation_id_deep_dot_notation():
    body = {"a": {"b": {"c": "found-it"}}}
    result = extract_correlation_id(body, "a.b.c")
    assert result == "found-it"


def test_extract_correlation_id_missing_key():
    body = {"foo": "bar"}
    result = extract_correlation_id(body, "correlationId")
    assert result is None


def test_extract_correlation_id_missing_nested_key():
    body = {"meta": {}}
    result = extract_correlation_id(body, "meta.correlationId")
    assert result is None


def test_extract_correlation_id_non_dict_intermediate():
    body = {"meta": "not-a-dict"}
    result = extract_correlation_id(body, "meta.correlationId")
    assert result is None


# ── store_input / retrieve_input ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_and_retrieve():
    store = MockCorrelationStore()
    key = "agent01:corr-1"
    data = {"body": {"key": "value"}, "received_at": "2024-01-01T00:00:00"}

    await store_input(store, key, data, ttl_s=300)
    result = await retrieve_input(store, key)

    assert result == data


@pytest.mark.asyncio
async def test_retrieve_missing_key():
    store = MockCorrelationStore()
    result = await retrieve_input(store, "nonexistent:key")
    assert result is None


@pytest.mark.asyncio
async def test_store_then_delete():
    store = MockCorrelationStore()
    key = "agent01:corr-2"
    data = {"body": {}, "received_at": "2024-01-01T00:00:00"}

    await store_input(store, key, data, ttl_s=60)
    await store.delete(key)
    result = await retrieve_input(store, key)
    assert result is None
