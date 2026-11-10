# Weights

| File / model | Size | In this repo | Role |
|---|---|---|---|
| `yolo26x.pt` | ~114 MB | No (see below) | **Default** detector/tracker (`--model yolo26x.pt`). Official Ultralytics YOLO26 (COCO). |
| `weights/yolo26m.pt` | ~44 MB | Yes | Lighter option (`--model weights/yolo26m.pt`). Official Ultralytics YOLO26 (COCO). |
| `depth-anything/Depth-Anything-V2-Small-hf` | — | No | Monocular relative depth. Downloaded from Hugging Face by `transformers` on first run and kept in the Hugging Face cache. |

## Getting `yolo26x.pt`

`yolo26x.pt` exceeds GitHub's 100 MB per-file limit, so it is not committed. Two options:

1. **Automatic**: the default `--model yolo26x.pt` is an official Ultralytics model name. If the file is not found,
   Ultralytics downloads it on first run into the current working directory (internet access required).
2. **Manual**: download it from the assets of this repository's GitHub release, place it e.g. in this folder and pass
   the path: `--model weights/yolo26x.pt`.

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

- YOLO26 weights: [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 (an Enterprise License is
  available from Ultralytics). Redistributed unmodified.
- Depth Anything V2 Small: Apache-2.0 (the Base/Large/Giant variants use a non-commercial license; only Small is used
  here).
- Both YOLO weights are trained on COCO; only the classes person, bicycle, car, motorcycle, bus and truck are used.
