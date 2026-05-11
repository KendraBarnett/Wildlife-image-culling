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
echo "Pulling vision models (one-time download, ~20 GB total):"
echo "  - llama3.2-vision:11b   default for triage (best with the long structured prompt)"
echo "  - qwen2.5vl:7b          alternative triage / used for the Compare-2-images feature"
echo "  - minicpm-v             smallest, fastest fallback"
echo "All three are downloaded so you can switch via WC_TRIAGE_MODEL in run.sh."
echo

for model in "llama3.2-vision:11b" "qwen2.5vl:7b" "minicpm-v"; do
  echo "Pulling $model ..."
  ollama pull "$model" || {
    echo "WARNING: failed to pull $model. You can re-run this script or run 'ollama pull $model' manually."
  }
done

echo
echo "Setup complete."
echo "Next: ./scripts/run.sh"
