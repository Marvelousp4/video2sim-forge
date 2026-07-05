#!/usr/bin/env python3
"""Validate MANO hand keypoint trajectories before V2S2R retargeting."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate_hand_world(path: Path) -> dict:
    data = np.load(path)
    if data.ndim != 3 or data.shape[1:] != (21, 3):
        raise ValueError(f"Expected shape (T, 21, 3), got {data.shape}")
    if data.shape[0] < 2:
        raise ValueError("Expected at least two frames")
    if not np.isfinite(data).all():
        raise ValueError("Trajectory contains NaN or Inf values")

    wrist = data[:, 0, :]
    step_dist = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    bbox_min = data.reshape(-1, 3).min(axis=0)
    bbox_max = data.reshape(-1, 3).max(axis=0)

    return {
        "path": str(path),
        "shape": list(data.shape),
        "frames": int(data.shape[0]),
        "keypoints": 21,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "wrist_step_mean": float(step_dist.mean()),
        "wrist_step_max": float(step_dist.max()),
        "dtype": str(data.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a V2S2R-compatible MANO right-hand world-frame .npy file."
    )
    parser.add_argument("--hand-world-npy", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, help="Optional path to write a JSON summary.")
    args = parser.parse_args()

    summary = validate_hand_world(args.hand_world_npy)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
