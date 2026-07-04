# Ubuntu GPU Proof Run

This directory contains sanitized proof from a real Ubuntu GPU run of the
video-to-sim pipeline on a tabletop bowl-and-fruit capture.

## What Was Tested

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5090
- SAM3 conda environment: CUDA available
- SAM3D objects conda environment: CUDA available
- SAM3 checkout: `757bbb0206a0b68bee81b17d7eb4877177025b2f`
- SAM3D checkout: `afdf6a31522d038c44c68a0bb57aa68827380797`
- Pipeline steps completed: SAM3 masks, SAM3D meshes/poses, scene assembly,
  world-frame export, visualization, and OBJ-to-URDF export

Step 1 Gemini was skipped for this proof run by reusing the same capture's
existing `gemini_scene.json`. A live Gemini `auto-pro` probe returned HTTP 400
on this machine, so the GPU proof focuses on the robotics reconstruction/export
chain rather than the API call.

## Reproduction Shape

The checked-in config keeps placeholder paths so private local capture paths are
not committed:

```bash
python scripts/validate_config.py --config examples/ubuntu_demo/config.yaml
python run_pipeline.py --config examples/ubuntu_demo/config.yaml
```

For an actual local rerun, copy the config and replace `input_dir`,
`output_dir`, and `camera_frame_json` with an approved RGB-D capture.

## Included Outputs

```text
outputs/
├── environment.md
├── validate_config.log
├── run.log
├── gemini_scene.json
├── mask_000.png
├── mask_001.png
├── mask_to_prompt_mapping.json
├── sam3d_results.json
├── scene_output.json
├── scene_output_new.json
├── scene_output_final.json
├── pipeline_timing.json
├── final_scene_visualization.png
└── urdfs/
    ├── obj_0000.urdf
    └── obj_0001.urdf
```

Large OBJ meshes are intentionally not committed:

- `obj_0000.obj`: about 21 MB
- `obj_0001.obj`: about 67 MB
- transformed meshes: about 19 MB and 60 MB

Those artifacts should be published through a GitHub Release or external
dataset if they are needed for full reproduction.

