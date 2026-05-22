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

"""DashboardReporter — decorator around IReporter that also feeds the dashboard.

The wrapped reporter is called first.  Dashboard updates are fire-and-forget:
if dashboard state is None (dashboard disabled) the overhead is a single
None-check per event — imperceptible to the agent processing loop.
"""

from sentinel.domain.models import HeartbeatEvent, ProcessedEvent
from sentinel.domain.ports.reporter import IReporter
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class DashboardReporter(IReporter):
    """Wraps another IReporter and mirrors every event into the dashboard state."""

    def __init__(self, inner: IReporter) -> None:
        self._inner = inner

    async def publish_event(self, event: ProcessedEvent) -> None:
        # Dashboard update first — cheap, never raises
        try:
            from sentinel.dashboard.state import get_state
            get_state().record_event(event)
        except Exception:
            pass  # dashboard must never affect agent processing

        await self._inner.publish_event(event)

    async def publish_heartbeat(self, heartbeat: HeartbeatEvent) -> None:
        try:
            from sentinel.dashboard.state import get_state
            get_state().record_heartbeat(heartbeat)
        except Exception:
            pass

        await self._inner.publish_heartbeat(heartbeat)
