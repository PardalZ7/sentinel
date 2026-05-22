#!/usr/bin/env bash
set -e

# Detect LocalStack endpoint (WSL needs Windows host IP)
if grep -qi microsoft /proc/version 2>/dev/null; then
  HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' | head -1)
  echo "[start] WSL detected, using host IP: $HOST_IP"
  export LOCALSTACK_ENDPOINT="http://${HOST_IP}:4566"
else
  export LOCALSTACK_ENDPOINT="http://localhost:4566"
fi

echo "[start] Waiting for LocalStack at $LOCALSTACK_ENDPOINT ..."
for i in $(seq 1 40); do
  if curl -sf "$LOCALSTACK_ENDPOINT/_localstack/health" > /dev/null 2>&1; then
    echo "[start] LocalStack is ready."
    break
  fi
  if [ "$i" -eq 40 ]; then
    echo "[start] ERROR: LocalStack not ready after 40 attempts. Is Docker running?"
    exit 1
  fi
  sleep 2
done

echo "[start] Starting all apps..."
LOCALSTACK_ENDPOINT=$LOCALSTACK_ENDPOINT npx concurrently \
  "LOCALSTACK_ENDPOINT=$LOCALSTACK_ENDPOINT npm run start --workspace=apps/app01" \
  "LOCALSTACK_ENDPOINT=$LOCALSTACK_ENDPOINT npm run start --workspace=apps/app02" \
  "LOCALSTACK_ENDPOINT=$LOCALSTACK_ENDPOINT npm run start --workspace=apps/app03" \
  "LOCALSTACK_ENDPOINT=$LOCALSTACK_ENDPOINT npm run start --workspace=apps/app04" \
  "LOCALSTACK_ENDPOINT=$LOCALSTACK_ENDPOINT npm run start --workspace=dashboard"
