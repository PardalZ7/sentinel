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

from collections.abc import Callable
from typing import Any

import grpc
from grpc import aio

from sentinel.adapters.grpc.generated import sentinel_pb2, sentinel_pb2_grpc
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class SentinelCortexServicer(sentinel_pb2_grpc.SentinelCortexServicer):
    """gRPC servicer that routes incoming events to a handler callback."""

    def __init__(self, event_handler: Callable[[Any], Any]) -> None:
        self._event_handler = event_handler

    async def ReportEvent(
        self,
        request: sentinel_pb2.ProcessedEventProto,
        context: grpc.aio.ServicerContext,
    ) -> sentinel_pb2.Ack:
        logger.info(
            "grpc_report_event",
            agent=request.agent_name,
            correlation_id=request.correlation_id,
        )
        try:
            await self._event_handler(request)
            return sentinel_pb2.Ack(ok=True)
        except Exception as exc:
            logger.error("grpc_report_event_failed", error=str(exc))
            return sentinel_pb2.Ack(ok=False)

    async def ReportHeartbeat(
        self,
        request: sentinel_pb2.HeartbeatProto,
        context: grpc.aio.ServicerContext,
    ) -> sentinel_pb2.Ack:
        logger.info(
            "grpc_report_heartbeat",
            agent=request.agent_name,
        )
        try:
            await self._event_handler(request)
            return sentinel_pb2.Ack(ok=True)
        except Exception as exc:
            logger.error("grpc_report_heartbeat_failed", error=str(exc))
            return sentinel_pb2.Ack(ok=False)

    async def ReportTemporalHeartbeat(
        self,
        request: sentinel_pb2.TemporalHeartbeatProto,
        context: grpc.aio.ServicerContext,
    ) -> sentinel_pb2.Ack:
        logger.info("grpc_report_temporal_heartbeat", layer=request.layer_name)
        try:
            await self._event_handler(request)
            return sentinel_pb2.Ack(ok=True)
        except Exception as exc:
            logger.error("grpc_report_temporal_heartbeat_failed", error=str(exc))
            return sentinel_pb2.Ack(ok=False)

    async def ReportTemporalAnomalySignal(
        self,
        request: sentinel_pb2.TemporalAnomalySignalProto,
        context: grpc.aio.ServicerContext,
    ) -> sentinel_pb2.Ack:
        logger.info("grpc_report_temporal_anomaly_signal", layer=request.layer_name)
        try:
            await self._event_handler(request)
            return sentinel_pb2.Ack(ok=True)
        except Exception as exc:
            logger.error("grpc_report_temporal_anomaly_signal_failed", error=str(exc))
            return sentinel_pb2.Ack(ok=False)


def create_server(
    host: str,
    port: int,
    event_handler: Callable[[Any], Any],
) -> grpc.aio.Server:
    """Create an async gRPC server for the Cortex."""
    server = aio.server()
    servicer = SentinelCortexServicer(event_handler=event_handler)
    sentinel_pb2_grpc.add_SentinelCortexServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    logger.info("grpc_server_created", host=host, port=port)
    return server


async def start_server(server: grpc.aio.Server) -> None:
    """Start the async gRPC server and wait until termination."""
    await server.start()
    logger.info("grpc_server_started")
    await server.wait_for_termination()
