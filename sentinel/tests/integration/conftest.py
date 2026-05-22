import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def docker_compose_file():
    return "tests/integration/docker-compose.test.yml"


@pytest_asyncio.fixture
async def redis_client(docker_services):
    """Wait for Redis to be ready and return an aioredis client."""
    import redis.asyncio as aioredis

    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.5,
        check=lambda: _is_redis_ready(),
    )

    client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def localstack_endpoint(docker_services):
    """Wait for LocalStack to be ready and return the endpoint URL."""
    import httpx

    endpoint = "http://localhost:4566"

    docker_services.wait_until_responsive(
        timeout=60.0,
        pause=1.0,
        check=lambda: _is_localstack_ready(endpoint),
    )

    yield endpoint


def _is_redis_ready() -> bool:
    import socket

    try:
        with socket.create_connection(("localhost", 6379), timeout=1):
            return True
    except OSError:
        return False


def _is_localstack_ready(endpoint: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{endpoint}/_localstack/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False
