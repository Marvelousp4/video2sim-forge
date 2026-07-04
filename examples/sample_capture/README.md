# Sample Capture Placeholder

This directory documents the expected layout for a small public sample capture.
The actual RGB-D video, depth images, and generated outputs are intentionally
not committed yet.

Expected structure:

```text
sample_capture/
├── color_video.mp4
├── cam_K.txt
├── camera_frame_pose.json
├── depth/
│   └── 0.png
└── scene_capture/
    ├── image/
    │   └── 0.png
    └── depth/
        └── 0.png
```

Before the first public release, add a small synthetic or approved real sample
capture so contributors can run at least part of the pipeline.

