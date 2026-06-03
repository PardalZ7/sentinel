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

from abc import ABC, abstractmethod


class ICorrelationStore(ABC):
    @abstractmethod
    async def save(self, key: str, data: dict, ttl_s: int) -> None: ...

    @abstractmethod
    async def get(self, key: str) -> dict | None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def record_group_source(
        self,
        key: str,
        source_name: str,
        event_snapshot: dict,
        expected_sources: list[str],
        ttl_s: int,
    ) -> dict:
        """Atomically add one source's snapshot to a group and return the updated group.

        Creates the group if it does not exist. The read-modify-write must be
        atomic so that concurrent events from different sources do not overwrite
        each other.
        """
        ...

    @abstractmethod
    async def try_claim_complete_group(
        self,
        key: str,
        expected_sources: list[str],
    ) -> bool:
        """Atomically check if all sources are present and, if so, delete the key.

        Returns True if the group was complete and this call claimed (deleted) it.
        Returns False if the group is incomplete, missing, or already claimed by
        another concurrent caller. Only one concurrent caller will get True for
        a given key.
        """
        ...
