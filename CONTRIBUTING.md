# Contributing

Video2Sim Forge is early-stage robotics tooling. Contributions that improve
reproducibility, dataset compatibility, simulation export quality, and setup
documentation are especially welcome.

## Good First Contributions

- Add a small public sample capture or fixture.
- Improve error messages for missing SAM3, SAM3D, RealSense, or Gemini setup.
- Add unit tests around JSON assembly, transform math, and URDF physics export.
- Document successful environment setups across CUDA, macOS, and Linux.

## Development Loop

```bash
python -m compileall -q .
python -m pytest
python -m ruff check .
```

Some pipeline steps require external model repositories, GPU support, camera
hardware, or API credentials. If a change cannot be tested end-to-end, state
which partial checks you ran in the pull request.

## Pull Request Checklist

- Keep API keys, datasets, generated meshes, and model checkpoints out of git.
- Include a short description of the robotics scenario being improved.
- Add or update tests when changing output schema, transform math, or physics
  calculations.
- Update documentation when adding new dependencies, environment variables, or
  input/output files.

