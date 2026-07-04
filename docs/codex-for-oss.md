# Codex for OSS Positioning

This note summarizes why Video2Sim Forge is a credible Codex for Open Source
candidate and how Codex would be used in the maintainer workflow.

Ready-to-adapt application responses are in
[application-answers.md](application-answers.md).

## Project Scope

Video2Sim Forge focuses on the video-to-simulation asset pipeline:

1. analyze an RGB-D capture for objects, task context, and material hints
2. segment prompted objects
3. reconstruct object meshes and camera-frame poses
4. export world-frame scene JSON and transformed meshes
5. optionally export URDF assets with approximate physics metadata

The project stops at simulation-ready assets. Policy learning, sim training,
real-robot deployment, and benchmark automation are downstream consumers, not
part of the core repository.

## Why It Matters

Robotics teams often spend significant engineering time converting real-world
captures into simulator assets. This repository makes that conversion path more
inspectable and reproducible by separating the pipeline into auditable steps,
documenting input/output contracts, and keeping intermediate artifacts available
for debugging.

The likely users are robotics researchers, manipulation teams, warehouse
automation developers, and students who need a transparent bridge between
RGB-D captures and simulation scenes.

## How Codex Helps Maintenance

Codex is most useful here for recurring maintainer work:

- reviewing pull requests for output-schema regressions
- adding tests around pure Python assembly, transform, and export utilities
- triaging installation issues across Linux, CUDA, SAM3, SAM3D, and RealSense
- improving docs when dependency or capture-layout assumptions change
- checking security-sensitive changes for leaked keys, private captures, and
  generated scene assets
- drafting release notes from merged changes

## Evidence of Maintainability

The repository includes:

- a documented quickstart and dependency guide
- an input/output contract for captures and generated artifacts
- a sanitized proof fixture from a completed video-to-sim run
- issue and pull request templates
- GitHub Actions for compile, lint, and unit tests
- focused unit tests for model-independent pipeline logic
- security guidance for API keys and private RGB-D data

## Near-Term Roadmap

- publish a small redistributable raw capture for model-dependent reruns
- expand the proof fixture with approved meshes and URDFs
- expand tests for transform math and URDF physics metadata
- add clearer troubleshooting for SAM3/SAM3D environment setup
- move shared helpers out of scripts into importable modules
- add benchmark notes for common tabletop manipulation captures
