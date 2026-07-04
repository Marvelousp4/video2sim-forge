# Ubuntu GPU Demo Placeholder

This directory is reserved for the first public Ubuntu GPU end-to-end run.

Until an approved raw capture and model outputs are available, use
[examples/proof_run](../proof_run) for sanitized derived-output evidence and
fixture tests.

## Expected Workflow

```bash
python scripts/validate_config.py --config examples/ubuntu_demo/config.yaml
python run_pipeline.py --config examples/ubuntu_demo/config.yaml
```

The config currently points at placeholder paths. On the Ubuntu GPU machine,
copy it to a local file or update it to reference an approved capture.

## Evidence to Add After a Successful Run

```text
outputs/
├── gemini_scene.json
├── sam3d_results.json
├── scene_output.json
├── scene_output_new.json
├── scene_output_final.json
├── final_scene_visualization.png
└── pipeline_timing.json
```

Large videos, depth folders, meshes, and URDF archives should go in a GitHub
Release or external dataset unless they are small and approved for direct git
storage.

