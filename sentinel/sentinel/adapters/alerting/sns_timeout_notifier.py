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

from sentinel.domain.ports.timeout_notifier import ITimeoutNotifier
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class SnsTimeoutNotifier(ITimeoutNotifier):
    """Publishes correlation timeout events to an SNS topic."""

    def __init__(
        self,
        topic_arn: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        self._topic_arn = topic_arn
        self._region = region
        self._endpoint_url = endpoint_url
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                import aiobotocore.session
                self._session = aiobotocore.session.get_session()
            except ImportError as exc:
                raise RuntimeError(
                    "aiobotocore is required for SNS transport. "
                    "Install it with: pip install aiobotocore"
                ) from exc
        return self._session

    async def notify(self, engine_name: str, correlation_id: str) -> None:
        payload = json.dumps({"engine": engine_name, "correlation_id": correlation_id, "event": "timeout"})
        session = self._get_session()
        async with session.create_client(
            "sns",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        ) as client:
            await client.publish(TopicArn=self._topic_arn, Message=payload)

        logger.debug("timeout_notified", engine=engine_name, correlation_id=correlation_id)
