# fly_vo — просмотр видео и разметка

Локальный веб-инструмент. Три страницы:

```
http://127.0.0.1:8765/review    проверка предложений системы  <- основная
http://127.0.0.1:8765/          разметка поворотов вручную
```

## Запуск

```bash
cd /Users/artem/test_fly
./webapp/run.sh
```

Держи терминал открытым, остановить — `Ctrl-C`. Если порт занят:

```bash
pkill -f "webapp/server.py"
./webapp/run.sh 9000
```

При первом запуске скрипт один раз перекодирует исходный ролик в
`webapp/media/VID00001.mp4` (720p h264), чтобы браузер мог его играть и
перематывать.

---

## /review — проверка предложений системы

Это инвертированный порядок работы: **не ты ищешь повороты, а система их
предлагает, а ты говоришь «верно» или «нет».**

Система прогоняет всё видео своим оценщиком движения
(`fly_vo/content_motion.py`, тот самый, что совпал с твоими глазами на 6 из 6
проверенных окон) и предлагает список кандидатов. Для каждого она уже назвала
направление. Твоя работа — посмотреть окно и подтвердить или отклонить.

Проверка одного пункта занимает секунды:

```
клип крутится сам по кругу
жмёшь 1  ->  верно
жмёшь 2  ->  неверно, затем указываешь, что там на самом деле
```

Клавиши: `1` верно · `2` неверно · `←` `→` между пунктами · `space` пауза

Если окно непонятное — это тоже ответ: жми «неверно» и выбирай «непонятно».

Результаты сразу пишутся на диск в `data/p01r/turn_verdicts.json`, отдельно от
предложений системы: сохраняется и что предложила система, и что увидел ты. Если
система ошиблась, расхождение остаётся видимым в файле. Кнопка «экспорт csv»
пишет плоскую таблицу рядом.

### Откуда берутся предложения

```bash
PYTHONPATH=. .venv/bin/python scripts/p02_scan_video.py
```

Стадия 1 прогоняет фронтенд по всему ролику (~6 минут, результат кэшируется).
Стадия 2 ищет участки, где боковое движение велико, однонаправлено, когерентно и
длится не меньше 0.8 с. Повторный запуск берёт кэш и отрабатывает мгновенно:

```bash
PYTHONPATH=. .venv/bin/python scripts/p02_scan_video.py --only detect
```

Результат: `data/p01r/candidate_turns.json`

**Важно:** предложения — это не метки. Меткой становится только то, что подтвердил
человек. Предложения с высокой уверенностью системы всё равно можно и нужно
отклонять: оценщик проверен на 8 читаемых окнах и совпал в 7, то есть он ошибается
иногда.

---

## / — разметка вручную

Для случаев, когда хочешь отметить поворот, которого система не предложила, или
переопределить направление.

**turns** — два клика на поворот: `i` в начале, `o` в конце, кнопка **measure**,
затем **accept**. Направление определяет тот же оценщик, что и на странице
проверки. Если не согласен — переопределяешь вручную, и расхождение сохраняется.

**trajectory** — рисунок пути камеры сверху. Даёт не только знак, но и скорость
поворота и суммарный угол. Но требует около одного клика на секунду видео (≈1500
кликов на весь ролик), и точность руки ограничивает: проверено, что при кликах
через 0.34 с курс между соседними точками определяется дрожанием мыши, а не
движением. Поэтому для меток не используется — только для короткого участка, где
нужна скорость.

**existing** — уже принятые метки, только для чтения.

## Куда пишутся данные

```
data/p01r/candidate_turns.json      предложения системы (генерируются)
data/p01r/turn_verdicts.json        твои вердикты по предложениям
data/p01r/trajectory_annotations.json  ручная разметка и рисунок пути
data/p01r/turns_export.csv          плоские таблицы для скриптов
```

## Why it exists

Every accuracy figure produced so far was scored against labels that included at
least one confirmed mislabelled reference, and several of the remaining labels
were never checked against the footage. The measurement side is now frozen and
validated; the label side is the bottleneck. This tool is for fixing that by
watching the clip and marking what the camera actually does.

## What it does

**turns** — the normal workflow, and two clicks per turn.

```
press i where the turn starts
press o where it ends
press measure
```

The server runs the frozen estimator on exactly that window and reports which way the
camera turned, with a confidence from the field coherence. Press **accept** if it
matches what you saw; if it does not, override it by hand and the disagreement is
recorded alongside the measurement. Nothing has to be drawn, and the work does not
scale with the length of the clip.

This works because the estimator was checked against hand labels on readable windows
and agreed on 7 of 8, with coherence 0.65-1.11. Your job is to catch the wrong ones.

**trajectory** — trace the camera path by hand. Richer in principle: the shape gives
direction, turn rate and total angle at once. But it needs roughly one click per second
of video, which is about 1500 clicks for this 25-minute clip, and it cannot represent
standing still — the points pile up in one spot. Kept for short dense stretches where
you want the rate, not for marking a whole route.

Turn detection there uses the *rate* of heading change, not the heading itself. That
matters: after a 90° left turn the path keeps pointing left, so a straight continuation
still shows a heading of −90° and would be mistaken for a continuing turn.

**existing** — the labels already decided for this clip, read-only, so you can see what
is settled and avoid re-marking the same turn.

## The timeline fix — read this before trusting any timestamp

`VID00001.AVI` declares 45000 frames at 30 fps (1500 s) but contains 36841 decodable
frames. Two different problems followed from that, and both corrupted earlier work.

**Problem 1 — seeking.** OpenCV computes a seek target from the declared frame count,
so `CAP_PROP_POS_FRAMES` lands early, and the error grows with position: frame 200
landed on 165 (−1.2 s), frame 5000 on 4095 (−30 s), frame 36800 on 30133 (−222 s).
Everything that seeked was reading the wrong frames. Fixed by `fly_vo/video_reader.py`,
which reads sequentially and never seeks. Verified: it matches `ffmpeg` (what a browser
demuxes) with a mean pixel difference of 3.3, where seeking differed by 64.2.

**Problem 2 — the browser timeline.** The first transcode kept the source's timestamps,
so the MP4 ended up with 36841 frames spread over 1500 s, an average of 24.56 fps. The
browser therefore played the content at 0.82× and every timestamp was about 18% late,
drifting past 200 s by the end. Measured directly: browser `t=106.0` showed true content
`t=86.75` (predicted 86.78). The review page was seeking to those times, so the reviewer
was shown **different moments than the ones the system had measured**. Fixed by
transcoding with `-r 30` **before** `-i`, which reinterprets the source as contiguous
30 fps. The result is 36841 frames at a true 30 fps, duration 1228 s, and the browser now
matches the reader exactly: `t=106.0` → `t=106.00`, residual 0.00 s.

The server serves whichever file is correct (`serve_video_name()`), so the pages never
need editing when the clip is rebuilt.

### Consequence for the first review round

The 37 verdicts from that round were collected against the broken timeline, so they
describe windows the system had not measured. They are kept in
`data/p01r/turn_verdicts_INVALID_timing.json` as evidence and must not be used as labels.
The candidate windows themselves are fine: their times were computed by a sequential
scan in true content time, so the same list can be reviewed again as-is.

## Where the data goes

Saving writes straight into the project, so nothing has to be mailed:

```
data/p01r/trajectory_annotations.json    turns + measurements + trajectory + notes
data/p01r/turns_export.csv               flat table, one row per turn
```

Each turn records whether it came from the measurement, a manual override, or a manual
override that agrees with the measurement, so disagreements stay visible.

`download json` is there as a backup, and `import` restores a previous session.

## The measure endpoint

`POST /api/measure {"t0": .., "t1": ..}` runs `fly_vo/content_motion.measure_window` on
that window and returns the direction in camera terms, the coherence ratio, the
tracking reliability and a confidence. It is the same frozen definition used everywhere
else, so the number in the browser and the number in a script are the same number.

## Reading the trajectory back

```bash
PYTHONPATH=. .venv/bin/python scripts/p01y2_trajectory_turns.py
```

Turns the drawing into turn labels and checks them two ways:

- **A. turns read off the shape**, each compared against the frozen measurement on
  exactly that window.
- **B. drawn turn size vs measured rotation** over the same window, both signed so
  positive means the camera turned right. A per-frame comparison was tried and
  dropped: the measured rate is noise-dominated (per-frame SNR ≈ 1.8, sign flips
  around 20 times a minute), so agreeing with a smooth hand-drawn curve sample by
  sample is not a valid test. The accumulated turn is what the drawing actually
  asserts, so that is what gets compared.

## Which turns are worth marking

Two clean turns of each direction are worth far more than ten muddy ones. A clean
turn is a single sustained sideways slide in one direction, with no strong pitch
and no walking bob dominating. The earlier pass failed partly because the three
windows it used contained two distinct camera states at best: `left_65` and
`right_67` overlap and sit on the same physical turn, and `right_216` turned out
to be a pitch-down with no clean yaw at all.

## Keyboard

| key | action |
| --- | --- |
| `space` | play / pause |
| `←` `→` | one frame |
| `shift`+`←` `→` | one second |
| `i` / `o` | mark in / mark out |
| `1` `2` `3` `4` | camera left / right / forward / unclear |
| `s` | save |
| `del` | delete the selected turn |
