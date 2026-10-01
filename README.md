# MaleCNS Visual Odometry

**Whole-brain fly navigation from video** — MaleCNS v1.0 (166,700 neurons) via [flybrain](https://github.com/alextitonis/fly.ai).

```
video.mp4 → visual encoder → photoreceptors + optic lobe inject
         → MaleCNS v1.0 (full graph) → descending neurons
         → forward / lateral / yaw → path integration → x, y, θ
```

| Layer | What | Source |
|-------|------|--------|
| Connectome | MaleCNS v1.0 (brain + VNC) | [flybrain](https://pypi.org/project/flybrain/) |
| Dynamics | Leaky integrate-and-fire | fly.ai / Shiu et al. |
| Visual input | `eye_drive` + LC4/LPLC2/T4/T5 inject | our `visual_encoder.py` |
| Navigation readout | DNa02 / DNg100 / MDN descending spikes | our `motion_readout.py` |
| Heading | EPG ring attractor | our `heading.py` |

> **Not FlyVis.** The old FAFB/FlyWire pipeline is archived in `archive/flyvis/`.

## Setup

Requires **Python 3.10+** (`flybrain` does not install on 3.9).

```bash
cd /Users/artem/test_fly
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Brain files (~260 MB, once) — downloaded automatically on first run
export FLY_DATA=$(pwd)/data/malecns
```

## Quick start

**Probe which populations respond to your video** (step 4 of the plan):

```bash
python scripts/probe_populations.py /path/to/video.mp4 --duration 30
```

**Batch trajectory:**

```bash
python run.py /path/to/video.mp4 -o output/
```

**Live web UI** (http://127.0.0.1:8770):

```bash
export FLY_DATA=$(pwd)/data/malecns
python scripts/run_web.py
```

## Architecture notes

- MaleCNS gives **wiring**, not a perfect digital brain — neuron model is simplified LIF.
- Photoreceptor → lamina relay is weak in spiking models; we also inject into visual projection / looming types (same strategy as fly.ai `FeatureDetectors` and `accurate-fly-brain`).
- Trajectory readout is **hand-tuned descending-neuron decoder** (DNa02 steer, DNg100 forward, MDN back). Tune gains after `probe_populations.py`.
- Dopamine learning adapts readout scalars against a user-drawn teacher path.

## Citations

- Berg et al., *Cell* 2026 — MaleCNS v1.0 connectome
- Shiu et al., *Nature* 2024 — LIF parameters
- fly.ai / alextitonis — `flybrain` engine
