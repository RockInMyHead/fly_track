# MaleCNS Visual Odometry — Forensic Pipeline Map

## Control checkpoints (where things break)

```text
VIDEO
  ↓
[1] Муха видит движение?          → visual_encoder metrics: motion, pan_shift, rotation_deg
  ↓
[2] T4/T5 / looming получают inject? → inject[] strengths → pathway_spikes (optic_lobe, looming)
  ↓
[3] MaleCNS реагирует?            → n_active, visual_projection spikes/step
  ↓
[4] Descending neurons fire?      → steer_L/R, forward_L/R (usually SILENT on video)
  ↓
[5] L/R asymmetry for yaw?        → DNa02 steer_L − steer_R (≈0 for inject-only video)
  ↓
[6] Forward signal?               → DNg100 forward_L+R (≈0) OR pathway motion→vx fallback
  ↓
[7] motion_readout decodes?       → vx_body, vy_body, yaw_rate, last_source (pathway|descending)
  ↓
[8] Heading integration?          → ORB corner snap / arc mode / heading lock (streaming.py)
  ↓
[9] Path integration x,y,θ?       → body_to_world(vx, vy, heading) × dt
```

## Exact data flow (one brain step @ 50 Hz)

```text
frame (BGR, max 480px)
  → preprocessing: grayscale 96×72 + ORB on 480×360 flow gray
  → retina/photoreceptors: eye_drive[n_visual] = row-mean panorama (1-D luminance)
  → inject (direct current, NOT spiking T4/T5 model):
       LC4/LPLC2  ← loom_strength = f(motion, contrast, |pan_shift|)
       visual_projection ← 0.2 × mean_lum
       T4/T5/HS/VS ← 0.4×motion + 0.15×|pan| + 0.25×|rotation_deg|
  → MaleCNS v1.0 (flybrain FlyBrain, LIF, dt=20ms, 166700 neurons)
  → pathway spike counts: optic_lobe, visual_projection, looming, steering, forward, …
  → descending groups: steer_L/R, forward_L/R, backward_L/R (DNa02, DNg100, MDN)
  → motion_readout:
       IF descending activity ≥ 0.5 spikes/window → DNa02/DNg100 decode
       ELSE pathway fallback → motion energy → vx, pan/ORB → yaw, ORB ref → heading snap
  → heading: RingAttractorCompass (EPG-inspired bump)
  → path integration: x += vx_world×dt, y += vy_world×dt
```

## Known limitations (as of 2026-09)

| Layer | Status |
|-------|--------|
| Photoreceptor → lamina | Weak in LIF; eye_drive alone insufficient |
| T4/T5 direction selectivity | **Not modeled** — scalar inject, not directional motion |
| LC4/LPLC2 looming | Responds to frame diff + contrast |
| Descending neurons | **~0.1 spikes/step** on video inject — readout uses pathway fallback |
| Yaw from ORB | Hand-tuned corner snap (90° L-turn) + arc mode (gentle curves) |
| Dopamine learning | 6 scalar params per video — not neural training |

## Key files

| File | Role |
|------|------|
| `fly_vo/visual_encoder.py` | Frame → eye_drive + inject + ORB rotation |
| `fly_vo/malecns_engine.py` | flybrain wrapper, pathway/descending counts |
| `fly_vo/motion_readout.py` | DN decode + pathway fallback |
| `fly_vo/egomotion.py` | EMA smoothing |
| `fly_vo/streaming.py` | Heading snap/arc, path integration |
| `fly_vo/heading.py` | Ring attractor compass |
| `fly_vo/config.py` | All gains and corner/arc thresholds |

## Readout source decision

```python
if descending_activity(window=5) >= 0.5:
    source = "descending"  # DNa02 steer, DNg100 forward
else:
    source = "pathway"     # motion→vx, ORB/pan→yaw, inject metrics
```

For VID00001 clips, **source is always `pathway`**.
