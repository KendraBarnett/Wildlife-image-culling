#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "No virtual environment found. Run ./scripts/setup.sh first."
  exit 1
fi

source .venv/bin/activate

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama is not running. Starting it..."
  if command -v brew >/dev/null 2>&1; then
    brew services start ollama || true
  fi
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

(
  sleep 2
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:8765" || true
  fi
) &

exec python -m wildlife_cull.server
