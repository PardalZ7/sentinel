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

# Start Node apps in background
node /app/apps/app01/index.js &
node /app/apps/app02/index.js &
node /app/apps/app03/index.js &
node /app/apps/app04/index.js &
node /app/dashboard/server.js &

# Register topology with sentinel control plane in background (non-blocking).
# Engines start regardless; the deploy only updates the control-plane view.
if [ -n "$SENTINEL_URL" ]; then
    (
        until sentinel deploy --sentinel-url "${SENTINEL_URL}" --no-engines 2>&1; do
            echo "[entrypoint] sentinel deploy failed, retrying in 5s..."
            sleep 5
        done
        echo "[entrypoint] topology registered with sentinel."
    ) &
fi

# Start correlation engines (foreground — keeps container alive)
exec sentinel start --config sentinel.json --mode engine
