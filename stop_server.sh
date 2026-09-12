#!/usr/bin/env bash
# Stop the dev server started by start_server.sh
set -euo pipefail

PID_FILE="/tmp/ellm-server.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found at $PID_FILE — server not running?"
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping server (PID $PID)..."
    kill "$PID"
    sleep 1
    
    # Check if it actually stopped
    if kill -0 "$PID" 2>/dev/null; then
        echo "Server didn't stop gracefully, sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Server stopped."
else
    echo "Process $PID not running (stale PID file)."
fi

rm -f "$PID_FILE"
