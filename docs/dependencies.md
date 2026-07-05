# Dependency Guide

Video2Sim Forge combines lightweight Python utilities with heavier robotics and
vision stacks. Install only the pieces you need.

## Base Pipeline

Use this for JSON assembly, frame extraction, transforms, visualization, and
URDF export:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Gemini Scene Analysis

Step 1 calls the Gemini REST API directly with `urllib`, so no provider SDK is
required. Export a local API key before running Step 1:

```bash
export GEMINI_API_KEY="..."
```

Do not commit real keys. Use `.env.example` as a template for local secrets.

## SAM3 Segmentation

Step 2 expects a separate `sam3` conda environment and a local SAM3 checkout or
package install.

```bash
conda create -n sam3 python=3.10
conda activate sam3
# Install PyTorch/CUDA for your machine, then install SAM3 following upstream docs.
export SAM3_ROOT=/path/to/sam3
```

`run_pipeline.py` launches this step with:

```bash
conda run -n sam3 --no-capture-output python scripts/step2_sam3.py ...
```

## SAM3D Reconstruction

Step 3 expects a separate `sam3d-objects` conda environment and a local SAM3D
objects checkout or package install.

```bash
conda create -n sam3d-objects python=3.10
conda activate sam3d-objects
# Install PyTorch/CUDA and SAM3D objects following upstream docs.
export SAM3D_ROOT=/path/to/sam-3d-objects
```

`run_pipeline.py` launches this step with:

```bash
conda run -n sam3d-objects --no-capture-output python scripts/step3_sam3d.py ...
```

## RealSense Capture

Camera capture utilities require Intel RealSense hardware and optional packages:

```bash
python -m pip install -r requirements-camera.txt
```

On macOS, `pyrealsense2` may require a platform-specific installation path. If
you only want to run the offline pipeline from existing RGB-D data, you do not
need RealSense dependencies.

## V2S2R Hand Trajectory Retargeting

Hand trajectory visualization is an optional downstream workflow. Install it in
a separate Ubuntu conda environment:

```bash
conda create -n v2s2r-retarget python=3.10 -y
conda activate v2s2r-retarget
python -m pip install -r requirements-retargeting.txt
```

Then clone V2S2R and run its retargeting scripts from that repository. See
[hand-trajectory-v2s2r.md](hand-trajectory-v2s2r.md) for the full bridge
workflow.

## Original Conda Exports

The `env_export/` directory contains development-machine conda exports for
reference. Prefer the instructions above for a clean public setup.
