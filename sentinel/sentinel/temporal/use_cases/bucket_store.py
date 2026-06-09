import json
from dataclasses import asdict

from sentinel.domain.models import SeasonalBucket


class BucketStore:
    def __init__(self, redis_client, layer_name: str, bucket_count: int) -> None:
        self._redis = redis_client
        self._layer_name = layer_name
        self._bucket_count = bucket_count

    def _key(self, index: int) -> str:
        return f"sentinel:temporal:{self._layer_name}:bucket:{index}"

    async def get_bucket(self, index: int) -> SeasonalBucket:
        raw = await self._redis.get(self._key(index))
        if raw is None:
            return SeasonalBucket()
        data = json.loads(raw)
        return SeasonalBucket(**data)

    async def save_bucket(self, index: int, bucket: SeasonalBucket) -> None:
        await self._redis.set(self._key(index), json.dumps(asdict(bucket)))

    async def get_all_buckets(self) -> list[SeasonalBucket]:
        return [await self.get_bucket(i) for i in range(self._bucket_count)]

    def bucket_confidence(self, buckets: list[SeasonalBucket]) -> int:
        if not buckets:
            return 0
        return min(b.n_observations for b in buckets)

    async def clear_all_buckets(self) -> None:
        keys = [self._key(i) for i in range(self._bucket_count)]
        if keys:
            await self._redis.delete(*keys)
