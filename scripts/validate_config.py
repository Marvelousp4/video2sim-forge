#!/usr/bin/env python3
"""Preflight checks for a Video2Sim Forge pipeline config."""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by CLI users without deps
    yaml = None


@dataclass
class Check:
    status: str
    name: str
    detail: str


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _has_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def _find_input_files(input_dir: Path):
    video = input_dir / "video.mp4"
    if not video.exists():
        video = input_dir / "color_video.mp4"

    depth = input_dir / "depth" / "0.png"
    if not depth.exists():
        depth = input_dir / "depth.mp4"
        if not depth.exists():
            depth = input_dir / "depth_video.mp4"

    intrinsics = input_dir / "cam_K.txt"
    if not intrinsics.exists():
        intrinsics = input_dir / "cam_params.txt"

    return video, depth, intrinsics


def _conda_env_exists(env_name: str) -> bool:
    if not shutil.which("conda"):
        return False
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(line.split() and line.split()[0] == env_name for line in result.stdout.splitlines())


def _torch_cuda_status():
    if importlib.util.find_spec("torch") is None:
        return False, "torch is not importable in this Python environment"

    code = (
        "import torch; "
        "print(torch.__version__); "
        "print(torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "torch CUDA probe failed"
    lines = result.stdout.strip().splitlines()
    available = len(lines) >= 2 and lines[1].strip() == "True"
    device = lines[2].strip() if len(lines) >= 3 and lines[2].strip() else "no CUDA device"
    version = lines[0].strip() if lines else "unknown torch"
    return available, f"torch {version}, {device}"


def _add(checks, status: str, name: str, detail: str):
    checks.append(Check(status=status, name=name, detail=detail))


def validate_config(config_path: Path, require_realsense: bool = False) -> list[Check]:
    if yaml is None:
        return [Check("ERROR", "PyYAML", "PyYAML is not installed")]

    checks: list[Check] = []
    config_path = config_path.resolve()
    if not config_path.exists():
        return [Check("ERROR", "config", f"Config file not found: {config_path}")]

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    input_dir = Path(config.get("input_dir", "")).expanduser()
    output_dir = Path(config.get("output_dir", "")).expanduser()
    camera_frame_json = config.get("camera_frame_json")

    skip_gemini = _as_bool(config.get("skip_gemini"), False)
    skip_sam3 = _as_bool(config.get("skip_sam3"), False)
    skip_sam3d = _as_bool(config.get("skip_sam3d"), False)
    skip_export = _as_bool(config.get("skip_export"), False)
    skip_obj_to_urdf = _as_bool(config.get("skip_obj_to_urdf"), False)

    if not input_dir:
        _add(checks, "ERROR", "input_dir", "input_dir is missing from config")
    elif input_dir.exists():
        _add(checks, "OK", "input_dir", str(input_dir))
    else:
        _add(checks, "ERROR", "input_dir", f"Directory not found: {input_dir}")

    if output_dir:
        _add(checks, "OK", "output_dir", str(output_dir))
    else:
        _add(checks, "ERROR", "output_dir", "output_dir is missing from config")

    if input_dir.exists():
        video, depth, intrinsics = _find_input_files(input_dir)
        _add(checks, "OK" if _has_file(video) else "ERROR", "RGB video", str(video))
        _add(checks, "OK" if _has_file(depth) else "ERROR", "depth input", str(depth))
        _add(checks, "OK" if _has_file(intrinsics) else "ERROR", "camera intrinsics", str(intrinsics))

        scene_image = input_dir / "scene_capture" / "image" / "0.png"
        scene_depth = input_dir / "scene_capture" / "depth" / "0.png"
        if not skip_sam3:
            _add(checks, "OK" if _has_file(scene_image) else "ERROR", "SAM3 image", str(scene_image))
        if not skip_sam3d:
            _add(checks, "OK" if _has_file(scene_image) else "ERROR", "SAM3D image", str(scene_image))
            _add(checks, "OK" if _has_file(scene_depth) else "ERROR", "SAM3D depth", str(scene_depth))

    if not skip_gemini:
        if os.environ.get("GEMINI_API_KEY"):
            _add(checks, "OK", "GEMINI_API_KEY", "set")
        else:
            _add(checks, "ERROR", "GEMINI_API_KEY", "not set")
    else:
        _add(checks, "SKIP", "GEMINI_API_KEY", "skip_gemini=true")

    if not skip_sam3:
        sam3_env = str(config.get("sam3_env", "sam3"))
        _add(checks, "OK" if _conda_env_exists(sam3_env) else "ERROR", "conda env SAM3", sam3_env)
        sam3_root = os.environ.get("SAM3_ROOT")
        _add(checks, "OK" if sam3_root and Path(sam3_root).exists() else "ERROR", "SAM3_ROOT", sam3_root or "not set")
    else:
        _add(checks, "SKIP", "SAM3", "skip_sam3=true")

    if not skip_sam3d:
        sam3d_env = str(config.get("sam3d_env", "sam3d-objects"))
        _add(checks, "OK" if _conda_env_exists(sam3d_env) else "ERROR", "conda env SAM3D", sam3d_env)
        sam3d_root = os.environ.get("SAM3D_ROOT")
        _add(checks, "OK" if sam3d_root and Path(sam3d_root).exists() else "ERROR", "SAM3D_ROOT", sam3d_root or "not set")
    else:
        _add(checks, "SKIP", "SAM3D", "skip_sam3d=true")

    if not skip_sam3 or not skip_sam3d:
        cuda_ok, cuda_detail = _torch_cuda_status()
        _add(checks, "OK" if cuda_ok else "ERROR", "PyTorch CUDA", cuda_detail)
    else:
        _add(checks, "SKIP", "PyTorch CUDA", "SAM3 and SAM3D are skipped")

    if not skip_export:
        if camera_frame_json:
            pose_path = Path(camera_frame_json).expanduser()
            _add(checks, "OK" if _has_file(pose_path) else "ERROR", "camera_frame_json", str(pose_path))
        else:
            _add(checks, "WARN", "camera_frame_json", "missing; export/URDF steps will be skipped or fail")
    else:
        _add(checks, "SKIP", "camera_frame_json", "skip_export=true")

    if skip_obj_to_urdf:
        _add(checks, "SKIP", "URDF export", "skip_obj_to_urdf=true")

    if require_realsense:
        if importlib.util.find_spec("pyrealsense2") is not None:
            _add(checks, "OK", "RealSense", "pyrealsense2 importable")
        else:
            _add(checks, "ERROR", "RealSense", "pyrealsense2 is not importable")
    else:
        _add(checks, "SKIP", "RealSense", "optional; pass --require-realsense to check")

    return checks


def print_report(checks: list[Check]):
    width = max((len(c.name) for c in checks), default=10)
    for check in checks:
        print(f"[{check.status:<5}] {check.name:<{width}}  {check.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Video2Sim Forge pipeline config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--require-realsense",
        action="store_true",
        help="Fail if pyrealsense2 is not importable",
    )
    args = parser.parse_args()

    checks = validate_config(Path(args.config), require_realsense=args.require_realsense)
    print_report(checks)
    return 1 if any(c.status == "ERROR" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())

