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
from collections.abc import AsyncIterator

from sentinel.domain.models import RawMessage


class ITransport(ABC):
    @abstractmethod
    def receive(self) -> AsyncIterator[RawMessage]: ...

    @abstractmethod
    async def ack(self, message_id: str) -> None: ...

    @abstractmethod
    async def nack(self, message_id: str) -> None: ...

    @abstractmethod
    async def publish(self, topic: str, payload: dict) -> None: ...
