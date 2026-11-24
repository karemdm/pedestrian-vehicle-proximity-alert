# How it works

The pipeline processes a video frame by frame. Each frame goes through relative
depth estimation, object tracking, rider and occupant filtering, a two-stage proximity gate
and a temporal smoothing step.

```mermaid
flowchart TD
    A["Video frame<br/>every Nth frame with --frame-step"] --> B["Depth Anything V2 Small<br/>frame converted BGR to RGB"]
    B --> C["Per-frame min-max normalize + invert<br/>z(x, y): 0 = nearest, 1 = farthest"]
    C --> R{"Depth map constant?<br/>(no depth evidence)"}
    R -- yes --> S["Frame skipped for proximity:<br/>nobody close, no window sample<br/>(warning logged once)"]
    A --> D["YOLO tracking<br/>person, bicycle, motorcycle (rider filter)<br/>car, bus, truck (targets)<br/>boxes clipped to the frame"]
    D --> E{"Person overlaps a bike / motorcycle<br/>≥ 50 % of its area?"}
    E -- yes --> F["Rider: remembered 2 s, excluded"]
    E -- no --> T{"Person ≥ 80 % inside a car / bus / truck box<br/>and at its depth (Δz ≤ 0.12)?"}
    T -- yes --> U["Occupant: remembered 2 s, excluded"]
    T -- no --> G["Pedestrian: foot point f, height h"]
    G --> H{"2D gate, per car / bus / truck:<br/>distance from f to the closest point c<br/>of the vehicle box ≤ 1.0 · h"}
    H -- no --> L
    H -- yes --> I{"Depth gate:<br/>Δz = abs(z(f) - z_vehicle) ≤ 0.12<br/>z_vehicle = z(c) if f is outside the box,<br/>else median of 3×3 interior points"}
    R -- no --> I
    I -- yes --> K["Close in this frame<br/>(any vehicle passes both gates)"]
    I -- no --> L["Not close"]
    K --> M["Sliding window per track<br/>1 s of video time"]
    L --> M
    M --> N{"Close now,<br/>or close ratio ≥ 50 %<br/>(at least one close sample)?"}
    N -- yes --> O["ALERT"]
    N -- no --> P["OK"]
```

## 1. Relative depth

Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) is run via the
`transformers` `depth-estimation` pipeline. The model predicts *relative inverse
depth* (disparity-like: larger = nearer). The pipeline's `depth` output `d` is that
prediction resized to the frame size and min-max scaled to an 8-bit image
(0 = farthest, 255 = nearest surface of the frame). It is then normalized again and
**inverted**, as in the original prototype:

```
z = 1 - (d - min(d)) / (max(d) - min(d))      (z = 1 everywhere if max - min <= 1e-5)
```

So **`z = 0` is the nearest and `z = 1` the farthest surface of the current frame**
(higher `z` = farther). This is what the `Z:` label shows; in the INFERNO depth PiP,
near surfaces are dark and far surfaces bright.

Properties of `z` to keep in mind:

- It is relative and per frame (re-scaled on every frame), not a distance in meters.
- It is an inverted *inverse* depth, so equal steps in `z` are not equal steps in
  distance: a given `Δz` corresponds to a much smaller metric gap near the camera
  than far away.
- It is quantized to 256 levels (the pipeline's 8-bit output), so the default
  `depth_threshold = 0.12` is about 31 levels.

The gates only use `|Δz|`, which is identical with or without the inversion, so the
inversion affects only the `Z` label and the PiP colors, not the alerts.

Depth at a point is the median of a square patch (7×7 by default, `--depth-patch`) around it, which
suppresses single-pixel noise and edge artifacts.

## 2. Detection and tracking

Ultralytics YOLO (`yolo26x.pt` by default, `imgsz=1280`, `conf=0.10`) is run with
`track(persist=True)`, restricted to COCO classes:

| Role | Classes |
|---|---|
| Person | person (0) |
| Rider vehicle | bicycle (1), motorcycle (3) |
| Target vehicle | car (2), bus (5), truck (7) |

Bicycles and motorcycles are used only to recognize riders (section 3) and are drawn in
cyan; they are **not** proximity targets, so a pedestrian next to one is never flagged.
Only cars, buses and trucks are checked in section 4.

Boxes are clipped to the frame. If the tracker returns no ids, the detection index
is used as id; such ids are not stable between frames and a warning is logged.

## 3. Rider filtering

A person is a **rider** in the current frame if any bicycle/motorcycle box covers at
least `rider_overlap` (0.50) of the person's box area:

```
overlap = area(person ∩ vehicle) / area(person)
```

The rider status is remembered for `rider_memory` (2.0 s) after the last overlap,
so short occlusions or box jitter do not turn a rider into a pedestrian. Riders are
drawn but never raise an alert. Memory entries older than `2 × rider_memory` are
removed.

### Occupants

A person is an **occupant** (driver or passenger) if at least `occupant_overlap` (0.80) of the person's box area lies
inside a car/bus/truck box *and* the person is at the vehicle's depth, `|z_person - z_vehicle| ≤ depth_threshold`,
where `z_vehicle` is the interior vehicle depth of section 4. Occupants reuse the rider mechanism (excluded, no
samples in the window, status remembered for `rider_memory`) but are tracked separately, drawn in magenta as
`Occupant #id`, and never raise an alert. The depth condition matters: a pedestrian standing in front of a bus has a
box entirely inside the bus box in the image, but a clear depth gap, so he or she stays a pedestrian.

## 4. Proximity gates (per pedestrian, per target vehicle)

Let `f = (x_f, y_f)` be the foot point (bottom-center of the person box) and `h`
the person box height (at least 10 px).

1. **2D gate**: `c` = closest point of the vehicle box to `f`.
   Pass if `‖f - c‖ ≤ height_factor × h` (default `1.0 × h`). Using the person height
   makes the radius scale with the apparent size of the person (perspective).
2. **Depth gate**: `Δz = |z(f) - z_vehicle|`. Pass if `Δz ≤ depth_threshold` (0.12). `z(f)` is the median patch at the
   foot. `z_vehicle` depends on where the foot is:
   - **Foot outside the vehicle box**: `z_vehicle = z(c)`, the median patch at the closest point of the box, i.e. the
     part of the vehicle nearest to the person (points on the frame border are sampled with the patch clipped to the
     image).
   - **Foot inside the vehicle box**: then `c = f`, so sampling there would return the person's own depth and `Δz`
     would always be 0. Instead `z_vehicle` is the median of `z` over a 3×3 grid of points spread over the box shrunk
     by 25 % on each side (the middle half of the box, where the body is rather than road, shadow or background).
     Grid points inside the person's box are ignored; if all are covered, all are used. The 25 % margin
     (`VEHICLE_INNER_MARGIN`) is a fixed constant, not a CLI option.

Depth-map exception: if the frame's depth map is constant (`max - min ≤ 1e-5` after normalisation, e.g. a black
frame) there is no depth evidence. `z` would be 1 everywhere and every 2D-gated pair would pass, so the frame is
skipped for proximity: nobody is close in it, the sliding windows get no sample, and a warning is logged once. The
8-bit depth input and the float `z` output are unchanged.

A pedestrian is *currently close* if any vehicle passes both gates. The smallest
`Δz` among vehicles passing the 2D gate is shown as `Delta Z`.

## 5. Temporal window and alert

Each pedestrian track keeps `(t, close)` samples from the last `time_window`
seconds (1.0 s of video time, `t = processed_frame_index / (fps / frame_step)`; the
window is inclusive, so it holds 31 samples at 30 FPS with `--frame-step 1`, and fewer
at higher steps). Then:

```
ratio = (# close samples) / (# samples in window)
ALERT  if  currently close  or  (ratio > 0 and ratio ≥ min_ratio (50 %))
```

Histories of tracks that have not been seen for longer than a grace period (the
time window, at least 1 s, in video time) are dropped; a track missing for a few
frames keeps its window. Riders add no samples while they are riders. The ratio mainly
keeps the alert on for a short time after the contact ends (hysteresis); an alert is
always raised while the person is currently close.

## Frame step and validation

With `--frame-step N` only frames 0, N, 2N, ... are decoded and processed (the others are
only skipped over); the output is written at `fps / N` and the time window, rider memory
and grace period are all measured on that effective frame rate, so seconds stay seconds.
Numeric parameters are validated before the run (CLI error or dialog in the setup window):
for example `--min-ratio` must be 0-100, `--pip-scale` between 0 and 1 (exclusive) and
`--frame-step` at least 1.

## 6. Rendering

- Target vehicles (cars, buses, trucks): blue boxes. Bicycles/motorcycles: cyan boxes.
- Riders: orange box, `Rider #id`. Occupants of vehicles: magenta box, `Occupant #id`.
- Pedestrians: green (OK) or red (ALERT) box, label `Person #id [STATUS (ratio%)]`
  and `Delta Z: … (Z: …)` (`N/A` when no vehicle passed the 2D gate); lines from the foot
  point to each vehicle contact point, yellow while OK and red during an ALERT.
- Colorized depth map `z` (INFERNO: near = dark, far = bright) with a white border as
  picture-in-picture, 25 % of the frame by default (`--pip-scale`), top-right.

## Parameter guide

| Parameter | Effect when increased |
|---|---|
| `--conf` | Fewer false detections, more missed small/occluded people. |
| `--imgsz` | Better small-object detection, slower. |
| `--height-factor` | Larger 2D radius → more candidate pairs, more alerts. |
| `--depth-threshold` | More tolerant depth gate → more alerts, more false positives from people behind/in front of vehicles. |
| `--time-window` | Alerts persist longer after the contact ends. |
| `--min-ratio` | Requires a larger share of close frames to keep an alert on. |
| `--rider-overlap` | Stricter rider detection → more riders treated as pedestrians. |
| `--occupant-overlap` | Stricter occupant detection → more people inside vehicle boxes treated as pedestrians. |
| `--rider-memory` | Rider or occupant status persists longer (also delays pedestrians who just got off a bike or out of a car). |
| `--frame-step` | Faster processing with fewer frames; tracking gets harder and short contacts may be missed. |

## Tuning tips

- Tune `--depth-threshold` per scene: since depth is normalized per frame, the same
  physical gap gives different `Δz` depending on the depth range visible in the frame
  (e.g. sky or a far background stretches the range).
- Check the `Delta Z` label on known close/not-close cases to pick a threshold.
- Start with `--height-factor 1.0`; raise it for top-down or distant cameras where people look small (the radius is `height_factor` times the person height).
- If IDs switch often, alerts flicker; a higher `--conf` or a larger model may help.
- Use `--no-display` on headless machines; the output video is still written.
