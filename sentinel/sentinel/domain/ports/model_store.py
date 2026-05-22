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
from typing import Any

from sentinel.domain.models import VersionMeta


class IModelStore(ABC):
    @abstractmethod
    async def save_version(self, name: str, model: Any) -> str: ...

    @abstractmethod
    async def load(self, name: str, version: str | None = None) -> Any: ...

    @abstractmethod
    async def list_versions(self, name: str) -> list[VersionMeta]: ...

    @abstractmethod
    async def rollback(self, name: str) -> None: ...
