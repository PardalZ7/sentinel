#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env.local if present
if [ -f "$SCRIPT_DIR/.env.local" ]; then
  set -a
  source "$SCRIPT_DIR/.env.local"
  set +a
  echo "[start] Loaded .env.local"
fi

# WSL: override LOCALSTACK_ENDPOINT with Windows host IP
if grep -qi microsoft /proc/version 2>/dev/null; then
  HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' | head -1)
  echo "[start] WSL detected, using host IP: $HOST_IP"
  export LOCALSTACK_ENDPOINT="http://${HOST_IP}:4566"
  export SQS_BASE_URL="http://${HOST_IP}:4566/${AWS_ACCOUNT_ID:-000000000000}"
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
npx concurrently \
  "npm run start --workspace=apps/app01" \
  "npm run start --workspace=apps/app02" \
  "npm run start --workspace=apps/app03" \
  "npm run start --workspace=apps/app04" \
  "npm run start --workspace=dashboard"
