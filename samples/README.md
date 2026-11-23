# Samples

| File | Content |
|---|---|
| `abbey_road_25s.mp4` | 25 s of the Abbey Road zebra crossing in London at night (1280×720, 30 FPS, no audio): pedestrians, cars, vans and a bus. |

## Usage

```bash
proximity-alert -i samples/abbey_road_25s.mp4 -o outputs/abbey_road.mp4 --model weights/yolo26m.pt --frame-step 2
```

## Source

Captured from the public Abbey Road crossing live webcam on YouTube: https://www.youtube.com/watch?v=zMCea32gpmg

People appear in a public space. The clip is included only to demonstrate the tool; if you appear in it and want it
removed, please open an issue.
