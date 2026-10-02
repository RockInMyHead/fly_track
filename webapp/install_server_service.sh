#!/bin/bash
# Держит webapp/server.py на :8765 через launchd (перезапуск при падении).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/webapp/com.fly_track.webapp.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.fly_track.webapp.plist"
UID_NUM="$(id -u)"
mkdir -p "$ROOT/output" "$HOME/Library/LaunchAgents"
sed "s|/Users/artem/test_fly|$ROOT|g" "$PLIST_SRC" > "$PLIST_DST"
launchctl bootout "gui/$UID_NUM/com.fly_track.webapp" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_DST"
launchctl enable "gui/$UID_NUM/com.fly_track.webapp"
sleep 1
if curl -sf -o /dev/null "http://localhost:8765/start5"; then
  echo "OK: http://localhost:8765/start5"
else
  echo "Сервис загружен, но HTTP пока не отвечает — см. $ROOT/output/webapp_server.log"
fi
echo "Остановить: launchctl bootout gui/$UID_NUM/com.fly_track.webapp"
