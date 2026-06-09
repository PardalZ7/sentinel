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
  - Consume N input queues (all streams — inputs and outputs of the monitored app)
  - Buffer partial captures in Redis (ICorrelationStore) until all N inputs arrive
  - When a group is complete, atomically claim it and publish a CorrelatedPair
  - Publish CorrelatedPair to one or more destination queues (IPairPublisher)

One CorrelationEngineRunner is instantiated per engine entry in sentinel.json.
Multiple engines run as concurrent asyncio tasks inside the same process.
"""

import asyncio
from dataclasses import dataclass

from sentinel.agent.use_cases import correlate
from sentinel.config.schema import CorrelationEngineConfig, InputConfig
from sentinel.domain.ports.correlation_store import ICorrelationStore
from sentinel.domain.ports.pair_publisher import IPairPublisher
from sentinel.domain.ports.transport import ITransport
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationEngineRunner:
    engine_config: CorrelationEngineConfig
    # Parallel list: one transport per input in engine_config.inputs
    input_transports: list[ITransport]
    correlation_store: ICorrelationStore
    pair_publishers: list[IPairPublisher]

    async def run(self) -> None:
        logger.info(
            "correlation_engine_started",
            engine=self.engine_config.name,
            num_inputs=len(self.engine_config.inputs),
            destinations=len(self.pair_publishers),
        )
        input_loops = [
            self._run_input_loop(inp_cfg, transport)
            for inp_cfg, transport in zip(self.engine_config.inputs, self.input_transports)
        ]
        await asyncio.gather(*input_loops)

    async def _run_input_loop(self, inp_cfg: InputConfig, transport: ITransport) -> None:
        mode = self.engine_config.effective_input_mode(inp_cfg)
        while True:
            batch = await transport.receive_batch()
            if not batch:
                continue
            await asyncio.gather(*[
                self._process_message(inp_cfg, transport, message, mode)
                for message in batch
            ])
            try:
                await transport.flush_acks()
            except Exception as exc:
                logger.error(
                    "engine_flush_acks_failed",
                    engine=self.engine_config.name,
                    input=inp_cfg.name,
                    error=str(exc),
                )

    async def _process_message(
        self, inp_cfg: InputConfig, transport: ITransport, message, mode: str
    ) -> None:
        try:
            if mode == "grouping":
                results = await correlate.store_input_captures_batch(
                    store=self.correlation_store,
                    config=self.engine_config,
                    input_cfg=inp_cfg,
                    message=message,
                )
                for cid, group in results:
                    await self._try_build_and_publish(cid, group)
            else:
                cid, group = await correlate.store_input_capture(
                    store=self.correlation_store,
                    config=self.engine_config,
                    input_cfg=inp_cfg,
                    message=message,
                )
                if cid is not None:
                    await self._try_build_and_publish(cid, group)
            await transport.ack(message.id)
        except Exception as exc:
            logger.error(
                "engine_input_error",
                engine=self.engine_config.name,
                input=inp_cfg.name,
                error=str(exc),
            )
            await transport.nack(message.id)

    async def _try_build_and_publish(self, correlation_id: str, group: dict) -> None:
        pair = await correlate.try_build_pair(
            store=self.correlation_store,
            config=self.engine_config,
            correlation_id=correlation_id,
            group=group,
        )
        if pair is not None:
            await self._publish_pair(pair)

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
