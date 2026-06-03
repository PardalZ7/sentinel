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

from sentinel.adapters.grpc.generated.correlated_pair_pb2 import CorrelatedPairProto, InputCaptureProto
from sentinel.domain.models import CorrelatedPair
from sentinel.domain.ports.pair_publisher import IPairPublisher
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def _to_proto(pair: CorrelatedPair) -> CorrelatedPairProto:
    inputs = [
        InputCaptureProto(
            name=inp.name,
            body=json.dumps(inp.body).encode(),
            received_at_ms=int(inp.received_at.timestamp() * 1000),
            size_bytes=inp.size_bytes,
        )
        for inp in pair.inputs
    ]
    return CorrelatedPairProto(
        engine_name=pair.engine_name,
        correlation_id=pair.correlation_id,
        inputs=inputs,
        output_body=json.dumps(pair.output_body).encode(),
        output_received_at_ms=int(pair.output_received_at.timestamp() * 1000),
        output_size_bytes=pair.output_size_bytes,
        processing_latency_ms=pair.processing_latency_ms,
        timed_out=pair.timed_out,
    )


class SqsPairPublisher(IPairPublisher):
    """Serializes a CorrelatedPair to protobuf and sends it to an SQS queue."""

    def __init__(
        self,
        queue_url: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        self._queue_url = queue_url
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
                    "aiobotocore is required for SQS transport. "
                    "Install it with: pip install aiobotocore"
                ) from exc
        return self._session

    async def publish(self, pair: CorrelatedPair) -> None:
        proto = _to_proto(pair)
        payload = proto.SerializeToString()

        session = self._get_session()
        async with session.create_client(
            "sqs",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        ) as client:
            import json as _json
            await client.send_message(
                QueueUrl=self._queue_url,
                MessageBody=_json.dumps({"_proto": payload.hex()}),
                MessageAttributes={
                    "ContentType": {
                        "DataType": "String",
                        "StringValue": "application/x-protobuf",
                    },
                    "EngineName": {
                        "DataType": "String",
                        "StringValue": pair.engine_name,
                    },
                },
            )
        logger.debug(
            "pair_published",
            engine=pair.engine_name,
            correlation_id=pair.correlation_id,
            queue=self._queue_url,
        )
