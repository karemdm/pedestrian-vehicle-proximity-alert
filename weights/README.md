# Weights

| File / model | Size | Where it comes from | Role |
|---|---|---|---|
| `yolo26x.pt` | ~114 MB | Official Ultralytics YOLO26 (COCO). Auto-downloaded by Ultralytics on first run; also attached to the GitHub release. Not stored in the repository because it exceeds GitHub's 100 MB file limit. | **Default** detector/tracker |
| `weights/yolo26m.pt` | ~43 MB | Official Ultralytics YOLO26 (COCO), shipped in this folder. | Lighter option (`--model weights/yolo26m.pt`) |
| `depth-anything/Depth-Anything-V2-Small-hf` | — | Downloaded from Hugging Face by `transformers` on first run and cached in the Hugging Face cache. | Monocular relative depth |

## Usage

```bash
# default: yolo26x.pt is downloaded automatically if not present
proximity-alert -i input.mp4 -o outputs/result.mp4

# lighter shipped model
proximity-alert -i input.mp4 -o outputs/result.mp4 --model weights/yolo26m.pt

# yolo26x.pt downloaded manually from the GitHub release into this folder
proximity-alert -i input.mp4 -o outputs/result.mp4 --model weights/yolo26x.pt
```

## Licenses

- YOLO26 weights: Ultralytics, AGPL-3.0 (enterprise license available from Ultralytics).
- Depth Anything V2 Small: Apache-2.0 (the Base/Large/Giant variants use a non-commercial license; only Small is used here).
- Both YOLO weights are trained on COCO; only the classes person, bicycle, car, motorcycle, bus and truck are used.
