#!/usr/bin/env bash
# For starting up server outside a docker container for quick tests. Not intended for proper deployment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
PID_FILE="/tmp/ellm-server.pid"
LOG_FILE="/tmp/uvicorn.log"
ERR_FILE="/tmp/uvicorn.err"
PORT=8008

# Load .env if present
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    echo "Loaded env from $ENV_FILE"
fi

# create default dir for db
DEFAULT_DB_DIR="$HOME/agent-home"
if [[ ! -d "$DEFAULT_DB_DIR" ]]; then
    mkdir -p "$DEFAULT_DB_DIR"
    echo "created $DEFAULT_DB_DIR"
fi
export AGENT_HOME_DB_PATH="$DEFAULT_DB_DIR/db.sqlite" # gets picked up by server

# Stop existing server if running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID $PID)..."
        kill "$PID"
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

# Start server
cd "$SCRIPT_DIR"
echo "Starting server... stdout: $LOG_FILE  stderr: $ERR_FILE"
nohup uv run uvicorn main:app --host 127.0.0.1 --port "$PORT" > "$LOG_FILE" 2> "$ERR_FILE" &
echo $! > "$PID_FILE"
echo "Server started (PID $(cat "$PID_FILE"))"

# Wait for health check
TIMEOUT=15
ELAPSED=0
echo "Waiting for server to become healthy..."
while [[ $ELAPSED -lt $TIMEOUT ]]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is healthy!"
        exit 0
    fi
    # Check if process died
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "ERROR: Server process died during startup!"
        echo "=== stderr ($ERR_FILE) ==="
        cat "$ERR_FILE"
        echo "=== stdout ($LOG_FILE) ==="
        cat "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo "ERROR: Server did not become healthy within ${TIMEOUT}s"
echo "=== stderr ($ERR_FILE) ==="
cat "$ERR_FILE"
echo "=== stdout ($LOG_FILE) ==="
cat "$LOG_FILE"
exit 1
