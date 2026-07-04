# Environment Exports

This directory keeps the original conda environment exports used during early
development. They are useful as a reference, but the preferred public setup path
is documented in the repository root `README.md`.

## Runtime Pieces

- `base` runs `run_pipeline.py` and Step 1 Gemini analysis.
- `sam3` runs Step 2 segmentation.
- `sam3d-objects` runs Step 3 reconstruction.
- `GEMINI_API_KEY` must be provided through your shell environment or a local
  `.env` file copied from `.env.example`.

## Files

| File | Use |
| --- | --- |
| `base.environment.yml` | Full base conda export from the development machine. |
| `sam3.environment.yml` | Full SAM3 conda export from the development machine. |
| `sam3d-objects.environment.yml` | Full SAM3D conda export from the development machine. |
| `*.from-history.yml` | Smaller explicit-install exports where available. |

Do not commit real API keys, capture data, model checkpoints, or generated
scene outputs.

