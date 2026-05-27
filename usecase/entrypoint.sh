#!/usr/bin/env bash
set -e

# Parse REDIS_URL (redis://user:pass@host:port) into SENTINEL_ vars
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
fi

# Derive SQS_BASE_URL from AWS env vars so sentinel.json ${SQS_BASE_URL} resolves correctly.
if [ -z "$SQS_BASE_URL" ]; then
    _region="${AWS_REGION:-us-east-1}"
    _account="${AWS_ACCOUNT_ID:-000000000000}"
    if [ -n "$LOCALSTACK_ENDPOINT" ]; then
        export SQS_BASE_URL="${LOCALSTACK_ENDPOINT}/${_account}"
    else
        export SQS_BASE_URL="https://sqs.${_region}.amazonaws.com/${_account}"
    fi
    echo "[entrypoint] SQS_BASE_URL derived: $SQS_BASE_URL"
fi

# Start Node apps in background
node /app/apps/app01/index.js &
node /app/apps/app02/index.js &
node /app/apps/app03/index.js &
node /app/apps/app04/index.js &

# Start correlation engines. If SENTINEL_URL is set, register topology first.
if [ -n "$SENTINEL_URL" ]; then
    (
        until sentinel deploy --sentinel-url "${SENTINEL_URL}" --no-engines 2>&1; do
            echo "[entrypoint] sentinel deploy failed, retrying in 5s..."
            sleep 5
        done
        echo "[entrypoint] topology registered with sentinel."
        echo "[entrypoint] starting correlation engines..."
        exec sentinel start --config sentinel.json --mode engine
    ) &
else
    echo "[entrypoint] SENTINEL_URL not set — starting correlation engines directly..."
    sentinel start --config sentinel.json --mode engine &
fi

# Dashboard runs in foreground — keeps the container alive and serves PORT
exec node /app/dashboard/server.js
