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

import asyncio
from datetime import datetime, timezone

import grpc
from grpc import aio

from sentinel.adapters.grpc.generated import sentinel_pb2, sentinel_pb2_grpc
from sentinel.domain.errors import GrpcUnavailableError
from sentinel.domain.models import HeartbeatEvent, ProcessedEvent
from sentinel.domain.ports.reporter import IReporter
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_MAX_BACKOFF_S = 60.0


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class GrpcReporter(IReporter):
    """IReporter implementation that publishes events via gRPC.

    Falls back to standalone (no-op) mode when the Cortex is unreachable.
    Retries with exponential backoff capped at _MAX_BACKOFF_S.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._standalone = False
        self._backoff_s = 1.0
        self._channel: grpc.aio.Channel | None = None
        self._stub: sentinel_pb2_grpc.SentinelCortexStub | None = None

    def _target(self) -> str:
        return f"{self._host}:{self._port}"

    def _get_stub(self) -> sentinel_pb2_grpc.SentinelCortexStub:
        if self._channel is None:
            self._channel = aio.insecure_channel(self._target())
            self._stub = sentinel_pb2_grpc.SentinelCortexStub(self._channel)
        return self._stub

    async def _try_connect(self) -> bool:
        """Attempt to get a healthy gRPC channel. Returns True on success."""
        try:
            stub = self._get_stub()
            # We rely on the first call to surface connectivity errors
            return True
        except Exception as exc:
            logger.warning(
                "grpc_connection_failed",
                target=self._target(),
                error=str(exc),
                backoff_s=self._backoff_s,
            )
            return False

    async def publish_event(self, event: ProcessedEvent) -> None:
        """Send a ProcessedEvent to the Cortex via gRPC."""
        if self._standalone:
            # Attempt reconnect with backoff
            connected = await self._try_connect()
            if not connected:
                logger.info("grpc_standalone_mode_event", agent=event.agent_name)
                return
            self._standalone = False
            self._backoff_s = 1.0

        proto = sentinel_pb2.ProcessedEventProto(
            agent_name=event.agent_name,
            correlation_id=event.correlation_id,
            input_received_at_ms=_to_ms(event.input_received_at),
            output_sent_at_ms=_to_ms(event.output_sent_at),
            processing_latency_ms=event.processing_latency_ms,
            anomaly_score=event.anomaly_score,
            is_anomaly=event.is_anomaly,
            anomaly_type=event.anomaly_type.value if event.anomaly_type else "",
            payload_schema_hash=event.payload_schema_hash,
            input_size_bytes=event.input_size_bytes,
            output_size_bytes=event.output_size_bytes,
        )

        try:
            stub = self._get_stub()
            await stub.ReportEvent(proto)
            self._backoff_s = 1.0
        except grpc.aio.AioRpcError as exc:
            logger.warning(
                "grpc_publish_event_failed",
                agent=event.agent_name,
                error=str(exc),
            )
            self._standalone = True
            self._channel = None
            self._stub = None
            await asyncio.sleep(min(self._backoff_s, _MAX_BACKOFF_S))
            self._backoff_s = min(self._backoff_s * 2, _MAX_BACKOFF_S)

    async def publish_heartbeat(self, heartbeat: HeartbeatEvent) -> None:
        """Send a HeartbeatEvent to the Cortex via gRPC."""
        if self._standalone:
            logger.info("grpc_standalone_mode_heartbeat", agent=heartbeat.agent_name)
            connected = await self._try_connect()
            if not connected:
                return
            self._standalone = False
            self._backoff_s = 1.0

        proto = sentinel_pb2.HeartbeatProto(
            agent_name=heartbeat.agent_name,
            timestamp_ms=_to_ms(heartbeat.timestamp),
            message_rate_per_sec=heartbeat.message_rate_per_sec,
            error_rate_last_window=heartbeat.error_rate_last_window,
            last_message_timestamp_ms=_to_ms(heartbeat.last_message_timestamp),
            model_phase=heartbeat.model_phase.value,
        )

        try:
            stub = self._get_stub()
            await stub.ReportHeartbeat(proto)
            self._backoff_s = 1.0
        except grpc.aio.AioRpcError as exc:
            logger.warning(
                "grpc_publish_heartbeat_failed",
                agent=heartbeat.agent_name,
                error=str(exc),
            )
            self._standalone = True
            self._channel = None
            self._stub = None
            await asyncio.sleep(min(self._backoff_s, _MAX_BACKOFF_S))
            self._backoff_s = min(self._backoff_s * 2, _MAX_BACKOFF_S)

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
