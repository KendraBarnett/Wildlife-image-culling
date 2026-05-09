#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3.12 >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "Python is not installed. Run: brew install python@3.12"
  exit 1
fi

PYTHON="$(command -v python3.12 || command -v python3)"

if ! command -v exiftool >/dev/null 2>&1; then
  echo "exiftool not found. Run: brew install exiftool"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found. Run: brew install ollama && brew services start ollama"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip wheel >/dev/null

echo "Installing Python dependencies (this may take 5-10 minutes the first time)..."
pip install -e .

mkdir -p data data/previews

echo
echo "Setup complete."
echo "Next: ./scripts/run.sh"
