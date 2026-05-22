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

import dataclasses
import json
from datetime import datetime

from sentinel.domain.models import Alert
from sentinel.domain.ports.alert_sink import IAlertSink
from sentinel.domain.ports.transport import ITransport
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def _alert_to_dict(alert: Alert) -> dict:
    """Serialize an Alert dataclass to a plain dict for transport."""
    d = dataclasses.asdict(alert)
    # Convert datetime to ISO string
    for key, value in d.items():
        if isinstance(value, datetime):
            d[key] = value.isoformat()
    return d


class QueueAlertSink(IAlertSink):
    """IAlertSink that publishes alerts to a transport topic."""

    def __init__(self, transport: ITransport, alarm_topic: str) -> None:
        self._transport = transport
        self._alarm_topic = alarm_topic

    async def send_alert(self, alert: Alert) -> None:
        """Publish the alert as a dict payload to the alarm topic."""
        payload = _alert_to_dict(alert)
        logger.info(
            "sending_alert",
            alert_type=alert.alert_type,
            severity=alert.severity,
            source=alert.source,
            affected_agents=alert.affected_agents,
        )
        await self._transport.publish(self._alarm_topic, payload)
