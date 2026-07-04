# Ubuntu GPU Setup

This guide is for a full video-to-sim run on Ubuntu with an NVIDIA GPU. It is
not required for docs-only work, fixture tests, or schema review.

## 1. Record the Machine

Save the environment facts before installing or running the pipeline:

```bash
uname -a
lsb_release -a
nvidia-smi
nvcc --version || true
python --version
conda --version
```

## 2. Clone and Install Base Dependencies

```bash
git clone git@github.com:Marvelousp4/video2sim-forge.git
cd video2sim-forge
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

Run the model-independent checks:

```bash
python -m compileall -q .
python -m ruff check .
python -m pytest
```

## 3. Verify PyTorch CUDA

Install the PyTorch build that matches your CUDA driver and machine. Then record:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 4. Install SAM3 and SAM3D

Create the conda environments named in your config:

```bash
conda create -n sam3 python=3.10
conda create -n sam3d-objects python=3.10
```

Install SAM3 and SAM3D objects according to their upstream instructions, then
export their checkout paths:

```bash
export SAM3_ROOT=/path/to/sam3
export SAM3D_ROOT=/path/to/sam-3d-objects
git -C "$SAM3_ROOT" rev-parse HEAD
git -C "$SAM3D_ROOT" rev-parse HEAD
```

## 5. Prepare a Local Config

```bash
cp config.example.yaml config.local.yaml
```

Edit `config.local.yaml` so `input_dir`, `output_dir`, and
`camera_frame_json` point at your approved RGB-D capture.

Set your API key without committing it:

```bash
export GEMINI_API_KEY="..."
```

Validate before the expensive run:

```bash
python scripts/validate_config.py --config config.local.yaml
```

## 6. Run and Save Evidence

```bash
mkdir -p runs/demo_ubuntu_gpu
python run_pipeline.py --config config.local.yaml | tee runs/demo_ubuntu_gpu/run.log
```

For a public proof commit, prefer sanitized small outputs:

```text
examples/ubuntu_demo/README.md
examples/ubuntu_demo/config.yaml
examples/ubuntu_demo/outputs/gemini_scene.json
examples/ubuntu_demo/outputs/sam3d_results.json
examples/ubuntu_demo/outputs/scene_output.json
examples/ubuntu_demo/outputs/scene_output_new.json
examples/ubuntu_demo/outputs/scene_output_final.json
examples/ubuntu_demo/outputs/final_scene_visualization.png
examples/ubuntu_demo/outputs/pipeline_timing.json
```

Keep large or private raw captures, full depth folders, meshes, and model
checkpoints out of git unless they have been explicitly approved and licensed.

