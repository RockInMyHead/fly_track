#!/usr/bin/env bash
# Запуск инструмента разметки.
#
#   ./webapp/run.sh                      # VID00001, первый банк, порт 8765
#   ./webapp/run.sh 9000                 # другой порт
#   REVIEW_ROUND=2 ./webapp/run.sh       # VID00001, второй банк
#   REVIEW_TAG=vid2 REVIEW_VIDEO=VID00002.AVI ./webapp/run.sh   # второе видео
#
# REVIEW_TAG задаёт суффикс файлов банка (review_set_<тег>.json,
# turn_verdicts_<тег>.json), поэтому разные клипы и раунды не перетирают друг друга.
# Без тега используется пара первого клипа.
#
# Перекодирует исходный ролик один раз, чтобы браузер мог его играть, затем
# запускает сервер и открывает страницу.

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-8765}"
SRC="data/p01r/VID00001.AVI"
OUT="webapp/media/VID00001_fixed.mp4"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

mkdir -p webapp/media

if [ ! -f "$OUT" ]; then
  if [ ! -f "$SRC" ]; then
    echo "исходный ролик не найден: $SRC"
    exit 1
  fi
  echo "перекодирую для браузера (один раз, несколько минут)..."
  # КРИТИЧНО: `-r 30` стоит ПЕРЕД `-i`. Исходный файл заявляет 45000 кадров, но
  # содержит 36841; без этой опции ffmpeg растягивает содержимое на 1500 секунд
  # (в среднем 24.56 fps), и тогда каждое время в браузере смещено примерно на
  # 18%, к концу ролика больше чем на 200 с. С `-r 30` перед входом источник
  # трактуется как непрерывные 30 fps, и шкала совпадает с содержимым.
  ffmpeg -y -loglevel error -r 30 -i "$SRC" -vf scale=1280:720 \
    -c:v libx264 -crf 28 -preset veryfast -pix_fmt yuv420p \
    -movflags +faststart -an "$OUT"
  echo "готово: $OUT"
fi

# контроль шкалы: должно быть 36841 кадр, 30/1, 1228 с
if command -v ffprobe >/dev/null; then
  echo -n "проверка шкалы: "
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=nb_frames,avg_frame_rate -show_entries format=duration \
    -of default=noprint_wrappers=1 "$OUT" | tr '\n' ' '
  echo
fi

echo "сервер: http://127.0.0.1:$PORT  (банк: REVIEW_ROUND=${REVIEW_ROUND:-1}, тег: ${REVIEW_TAG:-нет}, видео: ${REVIEW_VIDEO:-VID00001.AVI})"
( sleep 1; command -v open >/dev/null && open "http://127.0.0.1:$PORT/review.html" || true ) &
exec "$PY" webapp/server.py --port "$PORT"
