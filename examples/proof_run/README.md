# Proof Run: Bowl and Fruit

This directory contains a small derived-output fixture from a completed
Video2Sim-style RGB-D capture. It is included to prove the public pipeline shape
without committing raw camera videos, depth streams, private workspace captures,
or model checkpoints.

## Scenario

- task: interactive object manipulation
- manipulated object: orange fruit
- target object: light blue bowl
- reconstructed objects: 2
- exported frame: world/table frame from AprilTag calibration
- total recorded pipeline time: 93.728 seconds

## Included Artifacts

```text
outputs/
├── input_capture_preview.mp4
├── input_frame_first.png
├── input_frame_middle.png
├── input_frame_last.png
├── gemini_scene.json
├── sam3d_results.json
├── scene_output.json
├── scene_output_new.json
├── scene_output_final.json
├── pipeline_timing.json
└── final_scene_visualization.png
```

`input_capture_preview.mp4` and the three `input_frame_*.png` files show the
source tabletop manipulation: an orange fruit is moved into a light blue bowl.
`scene_output.json` shows the assembled camera-frame scene. `scene_output_new.json`
shows the same objects transformed into the calibrated table/world frame.
`scene_output_final.json` adds URDF paths plus approximate mass, inertia,
friction, and volume metadata for simulation import.

The JSON files were sanitized so local filesystem paths are replaced with
`<proof_run_input_dir>` and `<proof_run_output_dir>`.

## Not Included

Depth frame directories, masks, meshes, and URDF files are not included in this
public fixture yet. They are useful for full end-to-end reproduction, but they
need a licensing/privacy review before being published.

For now, this fixture supports docs, schema tests, maintainer review, and
application evidence for the video-to-sim asset path.
