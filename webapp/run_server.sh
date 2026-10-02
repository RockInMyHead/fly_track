#!/bin/bash
# Локальный сервер /start и /start5. Оставьте это окно терминала открытым.
# Надёжнее: ./webapp/install_server_service.sh (launchd, KeepAlive).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PORT="${1:-8765}"
echo "Сервер: http://localhost:${PORT}/start5"
echo "Остановка: Ctrl+C"
while true; do
  .venv/bin/python webapp/server.py --port "$PORT" || true
  echo "Сервер упал, перезапуск через 2 с…" >&2
  sleep 2
done
