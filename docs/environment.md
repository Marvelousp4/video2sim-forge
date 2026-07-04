# Environment Matrix

Video2Sim Forge has a small base Python layer and heavier optional robotics /
vision layers. Keep those boundaries clear when reporting issues or reviewing
pull requests.

## What Runs Anywhere

These checks should run on macOS, Linux CPU, and Ubuntu GPU machines:

```bash
python -m compileall -q .
python -m ruff check .
python -m pytest
python scripts/validate_config.py --config config.example.yaml
```

If `config.example.yaml` points at the placeholder sample capture, validation
will report missing capture files. That is expected until you point the config
at real RGB-D data or use a skip-heavy local test config.

## What Needs Ubuntu GPU

The model-dependent pipeline needs an Ubuntu machine with an NVIDIA GPU:

- Step 2 SAM3 text-guided segmentation
- Step 3 SAM3D object reconstruction
- full end-to-end RGB-D capture to simulation asset generation

The validator expects these pieces for a full run:

- `GEMINI_API_KEY` for Step 1 unless `skip_gemini: true`
- conda environment named by `sam3_env`
- conda environment named by `sam3d_env`
- `SAM3_ROOT` pointing at a SAM3 checkout
- `SAM3D_ROOT` pointing at a SAM3D objects checkout
- importable PyTorch with `torch.cuda.is_available() == True`
- capture files listed in [input-output.md](input-output.md)

## Evidence to Capture for Reproducibility

On the Ubuntu GPU machine, save these commands and outputs in the run log:

```bash
uname -a
lsb_release -a
nvidia-smi
nvcc --version || true
python --version
conda --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
echo "$SAM3_ROOT"
echo "$SAM3D_ROOT"
git -C "$SAM3_ROOT" rev-parse HEAD
git -C "$SAM3D_ROOT" rev-parse HEAD
```

Commit only sanitized, small outputs. Put large RGB-D videos, depth archives,
meshes, and model checkpoints in a GitHub Release or external dataset with a
clear license.

