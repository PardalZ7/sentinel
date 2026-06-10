#!/usr/bin/env bash
set -e

echo "[entrypoint] === starting usecase-webhook container ==="

# Parse REDIS_URL (redis://[user:pass@]host:port) into SENTINEL_ vars
if [ -n "$REDIS_URL" ]; then
    rest="${REDIS_URL#redis://}"
    if [[ "$rest" == *"@"* ]]; then
        userpass="${rest%@*}"
        hostport="${rest##*@}"
        SENTINEL_REDIS_PASSWORD="${userpass##*:}"
        export SENTINEL_REDIS_PASSWORD
    else
        hostport="$rest"
    fi
    SENTINEL_REDIS_HOST="${hostport%:*}"
    SENTINEL_REDIS_PORT="${hostport##*:}"
    export SENTINEL_REDIS_HOST
    export SENTINEL_REDIS_PORT
else
    echo "[entrypoint] WARNING: REDIS_URL not set — falling back to localhost:6379"
    export SENTINEL_REDIS_HOST="${SENTINEL_REDIS_HOST:-localhost}"
    export SENTINEL_REDIS_PORT="${SENTINEL_REDIS_PORT:-6379}"
fi
echo "[entrypoint] Redis: ${SENTINEL_REDIS_HOST}:${SENTINEL_REDIS_PORT}"

# Derive SQS_BASE_URL from AWS env vars so sentinel.json ${SQS_BASE_URL} resolves correctly.
if [ -z "$SQS_BASE_URL" ]; then
    _region="${AWS_REGION:-us-east-1}"
    _account="${AWS_ACCOUNT_ID:-000000000000}"
    if [ -n "$LOCALSTACK_ENDPOINT" ]; then
        export SQS_BASE_URL="${LOCALSTACK_ENDPOINT}/${_account}"
    else
        export SQS_BASE_URL="https://sqs.${_region}.amazonaws.com/${_account}"
    fi
fi
echo "[entrypoint] SQS_BASE_URL=${SQS_BASE_URL}"

# SNS base URL for sentinel.json variable resolution
if [ -z "$SNS_BASE_URL" ]; then
    _region="${AWS_REGION:-us-east-1}"
    _account="${AWS_ACCOUNT_ID:-000000000000}"
    if [ -n "$LOCALSTACK_ENDPOINT" ]; then
        export SNS_BASE_URL="${LOCALSTACK_ENDPOINT}"
    else
        export SNS_BASE_URL="https://sns.${_region}.amazonaws.com"
    fi
fi
echo "[entrypoint] SNS_BASE_URL=${SNS_BASE_URL}"

# Default output topic for the webhook
export DEFAULT_OUTPUT_TOPIC="${DEFAULT_OUTPUT_TOPIC:-ucwh-output}"
echo "[entrypoint] DEFAULT_OUTPUT_TOPIC=${DEFAULT_OUTPUT_TOPIC}"

# Start webhook server in background
echo "[entrypoint] starting webhook server..."
node /app/apps/webhook/index.js > /tmp/webhook.log 2>&1 &
WEBHOOK_PID=$!
echo "[entrypoint] webhook started (pid=$WEBHOOK_PID)"
sleep 1
if ! kill -0 $WEBHOOK_PID 2>/dev/null; then
    echo "[entrypoint] ERROR: webhook failed to start. Log:"
    cat /tmp/webhook.log
    exit 1
fi

# Start correlation engine immediately
echo "[entrypoint] starting correlation engine (mode=engine)..."
SENTINEL_LOG_FORMAT=pretty sentinel start --config sentinel.json --mode engine &
ENGINE_PID=$!
echo "[entrypoint] engine started (pid=$ENGINE_PID)"

# Register topology with sentinel control plane in background (non-blocking).
if [ -n "$SENTINEL_URL" ]; then
    echo "[entrypoint] SENTINEL_URL=${SENTINEL_URL} — registering topology..."
    (
        until sentinel deploy --sentinel-url "${SENTINEL_URL}" --no-engines 2>&1; do
            echo "[entrypoint] sentinel deploy failed, retrying in 5s..."
            sleep 5
        done
        echo "[entrypoint] topology registered with sentinel."
    ) &
else
    echo "[entrypoint] SENTINEL_URL not set — skipping topology registration"
fi

# Dashboard runs in foreground — keeps the container alive and serves PORT
echo "[entrypoint] starting dashboard..."
exec node /app/dashboard/server.js
