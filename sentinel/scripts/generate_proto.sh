#!/bin/bash
set -e
python -m grpc_tools.protoc \
  -I sentinel/adapters/grpc/proto \
  --python_out=sentinel/adapters/grpc/generated \
  --grpc_python_out=sentinel/adapters/grpc/generated \
  sentinel/adapters/grpc/proto/sentinel.proto

python -m grpc_tools.protoc \
  -I sentinel/adapters/grpc/proto \
  --python_out=sentinel/adapters/grpc/generated \
  sentinel/adapters/grpc/proto/correlated_pair.proto

echo "Proto stubs generated successfully."
