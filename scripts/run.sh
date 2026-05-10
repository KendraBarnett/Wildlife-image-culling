#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "No virtual environment found. Run ./scripts/setup.sh first."
  exit 1
fi

source .venv/bin/activate

# Phase 2 — on-demand scoring with Claude Sonnet 4.6.
# Uncomment and paste your key to enable the "Score with Claude" button.
# Without it, Phase 1 (local triage) still runs normally; only Phase 2 is gated.
# export ANTHROPIC_API_KEY="sk-ant-..."

# Hard cap on Claude spend. The UI warns at 80% and blocks new calls at
# 100%. Raise this number to allow more scoring; restart for it to take
# effect. Per-image cost on Sonnet 4.6 with prompt caching is roughly
# $0.005-$0.01, so $5 buys you ~500-1000 image scores.
export CLAUDE_BUDGET_USD="${CLAUDE_BUDGET_USD:-5.00}"

# Optional: pin to a specific model. Defaults to claude-sonnet-4-6.
# export CLAUDE_MODEL="claude-sonnet-4-6"

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
