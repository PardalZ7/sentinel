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
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from sentinel.domain.errors import TransportError
from sentinel.domain.models import RawMessage
from sentinel.domain.ports.transport import ITransport
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

# Maximum entries per SQS delete_message_batch call (AWS hard limit).
_SQS_BATCH_DELETE_MAX = 10


class SqsSnsTransport(ITransport):
    """AWS SQS/SNS transport using aiobotocore for async calls.

    Ack batching: receipt handles are queued and flushed in batches of up to 10
    (the SQS API limit) to reduce the number of delete_message API calls from N
    to ceil(N/10).  The batch is flushed automatically when it reaches the limit
    or explicitly via flush_acks().  Messages not yet flushed will be redelivered
    if the process crashes — SQS at-least-once delivery semantics are preserved.

    Client pooling: SQS and SNS clients are created once and reused across calls
    to avoid per-call TCP handshake overhead.  Call close() to release them.
    """

    def __init__(
        self,
        endpoint_url: str | None,
        input_queue_url: str,
        output_topic_arn: str,
        region: str = "us-east-1",
        wait_time_seconds: int = 1,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._input_queue_url = input_queue_url
        self._output_topic_arn = output_topic_arn
        self._region = region
        self._wait_time_seconds = wait_time_seconds
        self._session = None
        self._sqs_client = None
        self._sns_client = None
        self._receipt_handles: dict[str, str] = {}
        self._pending_acks: list[str] = []  # pending receipt handles for batch delete

    def _get_session(self):
        if self._session is None:
            try:
                import aiobotocore.session

                self._session = aiobotocore.session.get_session()
            except ImportError as e:
                raise TransportError(
                    "aiobotocore is required for SQS/SNS transport. "
                    "Install it with: pip install aiobotocore"
                ) from e
        return self._session

    async def _get_sqs_client(self):
        if self._sqs_client is None:
            session = self._get_session()
            self._sqs_client = await session.create_client(
                "sqs",
                region_name=self._region,
                endpoint_url=self._endpoint_url,
            ).__aenter__()
        return self._sqs_client

    async def _get_sns_client(self):
        if self._sns_client is None:
            session = self._get_session()
            self._sns_client = await session.create_client(
                "sns",
                region_name=self._region,
                endpoint_url=self._endpoint_url,
            ).__aenter__()
        return self._sns_client

    async def close(self) -> None:
        """Release persistent SQS/SNS clients."""
        if self._sqs_client is not None:
            try:
                await self._sqs_client.__aexit__(None, None, None)
            except Exception:
                pass
            self._sqs_client = None
        if self._sns_client is not None:
            try:
                await self._sns_client.__aexit__(None, None, None)
            except Exception:
                pass
            self._sns_client = None

    def _parse_sqs_message(self, msg: dict) -> RawMessage:
        """Parse a raw SQS message dict into a RawMessage, unwrapping SNS envelopes."""
        msg_id = msg["MessageId"]
        self._receipt_handles[msg_id] = msg["ReceiptHandle"]
        raw_body_str = msg["Body"]
        try:
            body = json.loads(raw_body_str)
            size_bytes = len(raw_body_str)
            was_sns = isinstance(body, dict) and body.get("Type") == "Notification" and "Message" in body
            if was_sns:
                inner_str = body["Message"]
                try:
                    body = json.loads(inner_str)
                    size_bytes = len(inner_str)
                except (json.JSONDecodeError, TypeError):
                    pass
            logger.debug("sqs_message_body", queue=self._input_queue_url, sns_unwrapped=was_sns, body_keys=list(body.keys()) if isinstance(body, dict) else type(body).__name__)
        except json.JSONDecodeError:
            body = {"raw": raw_body_str}
            size_bytes = len(raw_body_str)

        sent_ts_ms = int(msg.get("Attributes", {}).get("SentTimestamp", 0))
        received_at = (
            datetime.fromtimestamp(sent_ts_ms / 1000, tz=timezone.utc)
            if sent_ts_ms
            else datetime.now(tz=timezone.utc)
        )
        return RawMessage(id=msg_id, body=body, received_at=received_at, size_bytes=size_bytes)

    async def receive_batch(self) -> list[RawMessage]:
        """Poll SQS once and return up to 10 RawMessages. Returns empty list when queue is empty."""
        try:
            client = await self._get_sqs_client()
            response = await client.receive_message(
                QueueUrl=self._input_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=self._wait_time_seconds,
                AttributeNames=["SentTimestamp"],
            )
            messages = response.get("Messages", [])
            if messages:
                logger.info("sqs_messages_received", queue=self._input_queue_url, count=len(messages))
            return [self._parse_sqs_message(msg) for msg in messages]
        except Exception as exc:
            raise TransportError(f"SQS receive failed: {exc}") from exc

    async def receive(self) -> AsyncIterator[RawMessage]:
        """Poll SQS indefinitely, yielding one RawMessage at a time. Does NOT auto-ack."""
        while True:
            batch = await self.receive_batch()
            for raw_message in batch:
                yield raw_message

    async def ack(self, message_id: str) -> None:
        """Queue the message for batch deletion.

        The batch is flushed immediately when it reaches the SQS limit of 10.
        Call flush_acks() to force a flush of any remaining pending handles.
        """
        receipt_handle = self._receipt_handles.pop(message_id, None)
        if not receipt_handle:
            logger.warning("ack_missing_receipt_handle", message_id=message_id)
            return

        self._pending_acks.append(receipt_handle)
        if len(self._pending_acks) >= _SQS_BATCH_DELETE_MAX:
            await self.flush_acks()

    async def flush_acks(self) -> None:
        """Delete all pending ack'd messages in a single batch API call."""
        if not self._pending_acks:
            return

        batch, self._pending_acks = (
            self._pending_acks[:_SQS_BATCH_DELETE_MAX],
            self._pending_acks[_SQS_BATCH_DELETE_MAX:],
        )

        try:
            client = await self._get_sqs_client()
            await client.delete_message_batch(
                QueueUrl=self._input_queue_url,
                Entries=[
                    {"Id": str(i), "ReceiptHandle": h}
                    for i, h in enumerate(batch)
                ],
            )
        except Exception as exc:
            raise TransportError(f"SQS batch ack failed: {exc}") from exc

    async def nack(self, message_id: str) -> None:
        """No-op: message becomes visible again after visibility timeout."""
        self._receipt_handles.pop(message_id, None)
        logger.debug("nack_noop", message_id=message_id)

    async def publish(self, topic: str, payload: dict) -> None:
        """Publish a message to an SNS topic."""
        try:
            client = await self._get_sns_client()
            await client.publish(
                TopicArn=topic,
                Message=json.dumps(payload),
            )
        except Exception as exc:
            raise TransportError(f"SNS publish failed: {exc}") from exc
