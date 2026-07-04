# Changelog

## v0.1.0-alpha - 2026-07-04

Initial public alpha for Video2Sim Forge.

### Added

- Sanitized RGB-D video-to-simulation pipeline source.
- Gemini scene analysis, SAM3 segmentation, SAM3D reconstruction, transform,
  visualization, and OBJ-to-URDF export scripts.
- Public README, MIT license, contribution guide, security policy, issue
  templates, and pull request template.
- Dependency docs for base Python, RealSense, SAM3, SAM3D, and Ubuntu GPU setup.
- Environment matrix describing macOS, Linux CPU, Ubuntu GPU, and Windows/WSL2
  expectations.
- `scripts/validate_config.py` for preflight checks before expensive model runs.
- Sanitized proof fixtures:
  - `examples/proof_run` for derived JSON/images from a completed run.
  - `examples/ubuntu_demo` for an Ubuntu GPU proof run with environment notes,
    validator output, run log, masks, scene JSON, URDFs, timing, and final
    visualization.
- GitHub Actions CI for compile, lint, and unit tests.
- Unit tests for output assembly, proof fixtures, config validation, transform
  helpers, and URDF export behavior.
- Maintainer workflow documentation for review, release, and contribution
  practices.

### Validation

- `python -m compileall -q .`
- `python -m ruff check .`
- `python -m pytest`
- Ubuntu GPU proof run on Ubuntu 24.04.3 LTS with an NVIDIA GeForce RTX 5090.

### Known Limits

- Full model-dependent execution is expected to run on Ubuntu with an NVIDIA GPU.
- macOS is useful for docs, tests, config validation, and non-GPU utility work,
  but is not a supported full SAM3/SAM3D environment.
- The checked-in proof run uses sanitized derived artifacts. Large raw captures,
  OBJ meshes, transformed meshes, and model checkpoints are intentionally not
  committed.
- Step 1 Gemini was skipped in the Ubuntu GPU proof run by reusing an existing
  sanitized `gemini_scene.json`; future releases should add a fully
  redistributable raw capture and live scene-analysis repro path.
