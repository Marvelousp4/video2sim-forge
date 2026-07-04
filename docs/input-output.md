# Input and Output Contract

## Input Directory

Each capture should live in one directory:

```text
capture_name/
├── color_video.mp4          # or video.mp4
├── depth/                   # preferred
│   └── 0.png                # 16-bit depth PNG in millimeters
├── cam_K.txt                # or cam_params.txt
├── camera_frame_pose.json   # optional, required for world-frame export
└── scene_capture/
    ├── image/
    │   └── 0.png            # RGB frame used by SAM3/SAM3D
    └── depth/
        └── 0.png            # matching depth frame
```

`run_pipeline.py` currently uses the first `scene_capture/image/0.png` and
`scene_capture/depth/0.png` for segmentation and reconstruction.

## Camera Intrinsics

`cam_K.txt` may be a 3x3 matrix or a compact `[fx, fy, cx, cy]` style format,
depending on the downstream step. Keep units consistent with depth images.

## Camera Frame Pose

`camera_frame_pose.json` is used to transform object poses from the camera frame
to a tag/table/world frame. The current transform scripts assume AprilTag index
`0` is the reference frame.

## Main Outputs

```text
output_dir/
├── gemini_scene.json
├── mask_*.png
├── mask_to_prompt_mapping.json
├── obj_*.obj
├── sam3d_results.json
├── scene_output.json
├── scene_output_new.json
├── scene_output_final.json
├── transformed_meshes/
├── urdfs/
├── final_scene_visualization.png
├── pipeline_timing.txt
└── pipeline_timing.json
```

## Scene JSON Fields

`scene_output_final.json` is the simulation-facing artifact. Each object entry
includes:

- `object_id`
- `prompt`
- `material_type`
- `mesh_path`
- `pose.position_m`
- `pose.orientation_quat_xyzw`
- `is_manipulated`
- `urdf_path`
- mass, inertia, and friction metadata when URDF export succeeds

## Data Policy

Do not commit private RGB-D captures, customer site data, generated meshes, or
raw output directories. Keep shareable sample data small and clearly licensed.

