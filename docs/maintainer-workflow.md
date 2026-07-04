# Maintainer Workflow

Video2Sim Forge is intended to be maintained as an open robotics tool, not just
as a one-off research script dump. The maintained surface is the video-to-sim
asset pipeline: scene analysis, segmentation, reconstruction, transforms,
visualization, and optional URDF export.

Downstream policy training, benchmark automation, and real-robot deployment are
important consumers, but they are outside this repository's current scope.

## AI-Assisted Maintenance

AI coding assistants can help with:

- issue triage for installation, model setup, and input-format reports
- pull request review for transform math, output schema, and URDF export changes
- test generation for pure Python utilities
- documentation updates when dependencies or setup paths change
- release note drafting from merged commits
- security review for accidental API keys, private captures, or generated
  customer-site assets

Useful prompts:

```text
Review this PR for behavior regressions in scene JSON generation and URDF export.
Focus on bugs, missing tests, and backward-incompatible schema changes. Treat
robot training or deployment changes as out of scope unless they are docs-only.
```

```text
Given this issue log and the current README, draft a troubleshooting section for
SAM3/SAM3D environment setup without inventing unverified install commands.
```

```text
Add unit tests for this transform helper using small numeric fixtures. Do not
touch camera or model-dependent code.
```

## Release Checklist

- Run CI locally or confirm GitHub Actions passed.
- Confirm no API keys, private captures, generated meshes, or customer data are
  included.
- Update README and docs for any new environment variables or file formats.
- Add a short changelog entry describing user-visible changes.
- Tag releases only when the sample capture and documented setup have been
  checked.

## Project Areas

- `run_pipeline.py`: orchestration and output assembly
- `scripts/step1_gemini.py`: video-frame analysis and object prompt extraction
- `scripts/step2_sam3.py`: SAM3 segmentation
- `scripts/step3_sam3d.py`: SAM3D mesh reconstruction
- `scripts/step4_export_transforms.py`: camera-to-world transforms
- `scripts/step5_visualize.py`: PyVista scene rendering
- `scripts/step6_obj_to_urdf.py`: physics metadata and URDF export
- `camera/`: RealSense and AprilTag capture utilities
