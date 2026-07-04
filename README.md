# Scene Reconstruction Pipeline

A complete pipeline for reconstructing 3D scenes from RGB-D videos using Gemini, SAM3, and SAM3D.

## Pipeline Steps

1. **Step 1**: Gemini Scene Analysis
2. **Step 2**: SAM3 Segmentation 
3. **Step 3**: SAM3D Reconstruction
4. **Step 4**: Assemble Final Output
5. **Step 5**: Export Transformed Scene (poses → world frame, meshes with local edits)
6. **Step 6**: Visualize Transformed Scene
7. **Step 7**: OBJ to URDF Conversion (calculate mass, inertia, friction from mesh volume and material) 

## Requirements

- Python 3.8+
- Two conda environments: `sam3` and `sam3d-objects`
- GEMINI_API_KEY environment variable
- Input: RGB-D video (color_video.mp4, depth/*.png), camera intrinsics (cam_K.txt), camera poses (camera_frame_pose.json)

## Quick Start

1. **Configure the pipeline** - Edit `config.yaml`:
```yaml
input_dir: "input/Final_1"
output_dir: "output/Final_1"
camera_frame_json: "input/Final_1/camera_frame_pose.json"
```

2. **Run the pipeline**:
```bash
python run_pipeline.py --config config.yaml
```

## Output Files

```
output/Final_1/
├── gemini_scene.json              # Scene description from Gemini
├── masks/                         # Per-object segmentation masks
├── obj_*.obj                      # Original 3D meshes
├── scene_output.json              # Original poses (camera frame)
├── scene_output_new.json          # Transformed poses (world frame)
├── scene_output_final.json        # Final output with URDF paths, mass, inertia, friction
├── transformed_meshes/            # Meshes with local edits applied
│   └── obj_*_transformed.obj
├── urdfs/                         # URDF files with physics properties
│   └── obj_*.urdf
└── final_scene_visualization.png  # 3D visualization
```

## Physics Properties

The pipeline automatically calculates physics properties for each object:

- **Mass**: Calculated from mesh volume × material density
- **Inertia**: Approximated using bounding box dimensions
- **Friction**: Material-specific coefficients (static, dynamic, rolling, restitution)
- **Material Types**: metal, plastic, wood, cardboard, glass, ceramic, rubber, fabric

Material properties are obtained from Gemini's analysis and stored in `gemini_scene.json`.