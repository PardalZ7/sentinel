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

"""CorrelationEngineRunner — runs alongside the use case, not inside Sentinel cloud.

Responsibilities:
  - Consume N input queues and one output queue
  - Buffer partial captures in Redis (ICorrelationStore) until all N inputs arrive
  - Match complete groups against outputs to produce CorrelatedPair
  - Publish CorrelatedPair to one or more destination queues (IPairPublisher)
  - Notify via SNS on timeout (ITimeoutNotifier) — silently skipped if not configured

One CorrelationEngineRunner is instantiated per engine entry in sentinel.json.
Multiple engines run as concurrent asyncio tasks inside the same process.
"""

import asyncio
from dataclasses import dataclass, field

from sentinel.agent.use_cases import correlate
from sentinel.config.schema import CorrelationEngineConfig, InputConfig
from sentinel.domain.ports.correlation_store import ICorrelationStore
from sentinel.domain.ports.pair_publisher import IPairPublisher
from sentinel.domain.ports.timeout_notifier import ITimeoutNotifier
from sentinel.domain.ports.transport import ITransport
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationEngineRunner:
    engine_config: CorrelationEngineConfig
    # Parallel list: one transport per input in engine_config.inputs
    input_transports: list[ITransport]
    output_transport: ITransport
    correlation_store: ICorrelationStore
    pair_publishers: list[IPairPublisher]
    timeout_notifier: ITimeoutNotifier | None = field(default=None)

    async def run(self) -> None:
        logger.info(
            "correlation_engine_started",
            engine=self.engine_config.name,
            mode=self.engine_config.correlation_mode,
            num_inputs=len(self.engine_config.inputs),
            destinations=len(self.pair_publishers),
        )
        input_loops = [
            self._run_input_loop(inp_cfg, transport)
            for inp_cfg, transport in zip(self.engine_config.inputs, self.input_transports)
        ]
        await asyncio.gather(*input_loops, self._run_output_loop())

    async def _run_input_loop(self, inp_cfg: InputConfig, transport: ITransport) -> None:
        mode = self.engine_config.correlation_mode
        async for message in transport.receive():
            try:
                if mode == "grouping":
                    await correlate.store_input_captures_batch(
                        store=self.correlation_store,
                        config=self.engine_config,
                        input_cfg=inp_cfg,
                        message=message,
                    )
                else:
                    await correlate.store_input_capture(
                        store=self.correlation_store,
                        config=self.engine_config,
                        input_cfg=inp_cfg,
                        message=message,
                    )
                await transport.ack(message.id)
            except Exception as exc:
                logger.error(
                    "engine_input_error",
                    engine=self.engine_config.name,
                    input=inp_cfg.name,
                    error=str(exc),
                )
                await transport.nack(message.id)

    async def _run_output_loop(self) -> None:
        mode = self.engine_config.correlation_mode

        async for message in self.output_transport.receive():
            try:
                if mode == "grouping":
                    pairs = await correlate.resolve_pairs(
                        store=self.correlation_store,
                        config=self.engine_config,
                        message=message,
                    )
                else:
                    pair = await correlate.resolve_pair(
                        store=self.correlation_store,
                        config=self.engine_config,
                        message=message,
                    )
                    pairs = [pair] if pair is not None else []

                for pair in pairs:
                    if pair.timed_out:
                        await self._handle_timeout(pair.correlation_id)
                    else:
                        await self._publish_pair(pair)

                await self.output_transport.ack(message.id)
            except Exception as exc:
                logger.error(
                    "engine_output_error",
                    engine=self.engine_config.name,
                    error=str(exc),
                )
                await self.output_transport.nack(message.id)

    async def _publish_pair(self, pair) -> None:
        for publisher in self.pair_publishers:
            try:
                await publisher.publish(pair)
            except Exception as exc:
                logger.error(
                    "pair_publish_failed",
                    engine=self.engine_config.name,
                    correlation_id=pair.correlation_id,
                    error=str(exc),
                )

    async def _handle_timeout(self, correlation_id: str) -> None:
        logger.warning(
            "correlation_timeout",
            engine=self.engine_config.name,
            correlation_id=correlation_id,
        )
        if self.timeout_notifier is None:
            return
        try:
            await self.timeout_notifier.notify(self.engine_config.name, correlation_id)
        except Exception as exc:
            logger.warning(
                "timeout_notification_failed",
                engine=self.engine_config.name,
                correlation_id=correlation_id,
                error=str(exc),
            )
