# Codex for OSS Application Draft

These short answers are designed for the Codex for Open Source application
fields, which currently ask for concise maintainer, qualification, and API
credit details.

## Maintainer Role

Primary maintainer. I built and maintain Video2Sim Forge, an open-source
RGB-D video-to-simulation asset pipeline for robotics manipulation scenes. I
own the capture assumptions, Gemini/SAM3/SAM3D orchestration, scene-output
schemas, transform/export utilities, documentation, tests, CI, and public data
safety process.

## Why This Repository Qualifies

Video2Sim Forge makes RGB-D-to-simulation asset generation more reproducible for
robotics researchers, manipulation teams, warehouse automation developers, and
students. It turns real captures into object prompts, masks, meshes,
world-frame poses, URDFs, and approximate physics metadata. The public repo now
includes docs, CI, tests, security guidance, input/output contracts, config
validation, and sanitized proof artifacts from an Ubuntu GPU run.

## API Credit Use

API credits would support OSS maintenance: running scene-analysis tests on
approved sample captures, reproducing user issues, improving prompt/schema
robustness, validating new demo captures, and building Codex-assisted PR review
workflows for output-schema, transform, security, and documentation changes.

## Anything Else

The project is intentionally scoped to the video-to-sim asset path, not robot
policy training. That boundary keeps contributions reviewable without private
robot data or GPU-heavy runs, while still serving an important robotics
workflow: converting real-world captures into simulator-ready scenes. The
current alpha includes a small sanitized proof run and a roadmap toward a
redistributable raw capture, broader environment coverage, and cleaner package
boundaries.
