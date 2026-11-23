# Results

Annotated videos produced by this project.

- `abbey_road_proximity.mp4`: Zebra crossing at night (made with `--model weights/yolo26m.pt --frame-step 2`): pedestrians next to passing cars and vans, which are tracked as targets, with alert states and the depth picture-in-picture. A truck and a double-decker bus pass right next to the camera, partly out of frame, and are missed by `yolo26m` in most frames (no box, so no alert for them).
- `shinjuku_proximity.mp4`: Night-time intersection: pedestrians crossing near cars (many of them taxis) and trucks, with alert states and the depth picture-in-picture.

## Source footage

- `abbey_road_proximity.mp4`: captured from the public Abbey Road crossing live webcam on YouTube, https://www.youtube.com/watch?v=zMCea32gpmg
- `shinjuku_proximity.mp4`: a publicly available street recording.
- `docs/assets/demo.gif`: 6 s demo (720×406, 10 FPS) made from the same Shinjuku footage as `shinjuku_proximity.mp4`.
<!-- TODO(author): add the link of the Shinjuku source video. -->

People appear in public spaces; the outputs are shown only to illustrate the method.
If you appear in a video and want it removed, please open an issue.
