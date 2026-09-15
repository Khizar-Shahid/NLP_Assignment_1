#!/usr/bin/env bash
# Start the chatbot server. Assumes Ollama is already running and the model
# is pulled (see README). Set ENGINE=mock to run without a model.
set -e
export PORT="${PORT:-8000}"
echo "Starting on http://localhost:${PORT}  (engine=${ENGINE:-ollama}, model=${MODEL_NAME:-qwen2.5:3b-instruct-q4_K_M})"
uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --reload
