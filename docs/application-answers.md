# Codex for OSS Application Draft

These short answers are designed for the Codex for Open Source application
fields, which currently ask for concise maintainer, qualification, and API
credit details.

## Maintainer Role

Primary maintainer. I built and maintain the video-to-simulation pipeline,
including RGB-D capture assumptions, Gemini/SAM3/SAM3D orchestration,
scene-output schemas, transform/export utilities, documentation, tests, and
security/data handling for public release.

## Why This Repository Qualifies

Video2Sim Forge makes RGB-D-to-simulation asset generation more reproducible for
robotics researchers and manipulation teams. It turns real captures into object
prompts, masks, meshes, world-frame poses, URDFs, and physics metadata. The repo
now includes docs, CI, tests, security guidance, and a sanitized proof run.

## API Credit Use

API credits would support OSS maintenance: running scene-analysis tests on
approved sample captures, reproducing user issues, improving prompt/schema
robustness, and building Codex-assisted PR review workflows for output-schema,
transform, security, and documentation changes.

## Anything Else

The project is intentionally scoped to the video-to-sim asset path, not robot
policy training. This keeps contributions reviewable without private robot data
or GPU-heavy runs, while still serving an important robotics workflow: converting
real-world captures into simulator-ready scenes.

