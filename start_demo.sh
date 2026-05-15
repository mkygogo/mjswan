#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
LAN_IP="${LAN_IP:-192.168.3.38}"
DIST_DIR="examples/demo/dist"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not in PATH."
  exit 1
fi

if [[ "${1:-}" == "--rebuild" || ! -f "$DIST_DIR/index.html" ]]; then
  echo "[mjswan] Building demo..."
  MJSWAN_NO_LAUNCH=1 uv run --extra examples python examples/demo/main.py
else
  echo "[mjswan] Using existing $DIST_DIR"
fi

echo "[mjswan] Starting demo server..."
echo "[mjswan] Open: http://${LAN_IP}:${PORT}"

uv run --extra examples python - <<PY
from pathlib import Path
import mjswan

mjswan.mjswanApp(Path("${DIST_DIR}")).launch(
    host="${HOST}",
    port=int("${PORT}"),
    open_browser=False,
)
PY
