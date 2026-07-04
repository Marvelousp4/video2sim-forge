#!/usr/bin/env python3
"""
Scene Reconstruction Pipeline - Main Orchestration Script

This script runs the complete pipeline from RGBD video to reconstructed meshes:
1. Gemini Scene Analysis → Scene description with objects and material types
2. SAM3 Segmentation → Per-object masks
3. SAM3D Reconstruction → 3D meshes with poses

Requirements:
- GEMINI_API_KEY environment variable must be set
- Two conda environments: 'sam3' and 'sam3d-objects'
- Input directory with: video.mp4, depth.mp4, cam_K.txt

Usage:
    python run_pipeline.py --config config.yaml
    OR
    python run_pipeline.py --input_dir /path/to/task_folder --output_dir /path/to/output

Author: Pipeline Automation
"""

import os
import sys
import json
import time
import atexit
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_step(cmd: list, step_name: str, cwd: str = None, env: dict = None):
    """Run a pipeline step and handle errors."""
    log(f"Starting: {step_name}")
    log(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=False,  # Don't buffer - show output in real-time
            text=True,
            check=True
        )
        log(f"Completed: {step_name}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"FAILED: {step_name}", "ERROR")
        log(f"Return code: {e.returncode}", "ERROR")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def run_conda_step(conda_env: str, script_path: str, args: list, step_name: str):
    """Run a script in a specific conda environment."""
    log(f"Running {step_name} in conda env '{conda_env}'")
    
    # Build the conda run command
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", script_path
    ] + args
    
    log(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output stream through
            text=True,
            check=True
        )
        log(f"Completed: {step_name}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"FAILED: {step_name}", "ERROR")
        log(f"Return code: {e.returncode}", "ERROR")
        return False


def assemble_final_output(output_dir: Path) -> dict:
    """Assemble final scene_output.json from intermediate results."""
    
    # Load Gemini results
    gemini_path = output_dir / "gemini_scene.json"
    if not gemini_path.exists():
        raise FileNotFoundError(f"Gemini output not found: {gemini_path}")
    
    with open(gemini_path, 'r') as f:
        gemini_data = json.load(f)
    
    # Load SAM3D results
    sam3d_path = output_dir / "sam3d_results.json"
    if not sam3d_path.exists():
        raise FileNotFoundError(f"SAM3D output not found: {sam3d_path}")
    
    with open(sam3d_path, 'r') as f:
        sam3d_data = json.load(f)
    
    # # Load mask-to-prompt mapping (format: {mask_index: prompt_index})
    # mapping_path = output_dir / "mask_to_prompt_mapping.json"
    # if not mapping_path.exists():
    #     raise FileNotFoundError(f"Mask mapping not found: {mapping_path}")
    
    # with open(mapping_path, 'r') as f:
    #     mask_to_prompt_idx = json.load(f)


    # Load mask-to-prompt mapping.
    # Supports two formats:
    #   Legacy: {"0": 0, "1": 1}
    #   New:    {"0": {"prompt_idx": 0, "prompt": "...", "fallback_prompt": "...", "prompt_used": "..."}, ...}
    # Both are normalized to {mask_index_str: prompt_index_int}.
    mapping_path = output_dir / "mask_to_prompt_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mask mapping not found: {mapping_path}")

    with open(mapping_path, "r") as f:
        raw_mapping = json.load(f)

    def _coerce_prompt_idx(v):
        if isinstance(v, dict):
            return int(v["prompt_idx"])
        return int(v)

    mask_to_prompt_idx = {k: _coerce_prompt_idx(v) for k, v in raw_mapping.items()}
    
    # Get prompts list from Gemini
    gemini_objects = gemini_data.get("objects", [])
    manipulated_object = gemini_data.get("manipulated_prompt", gemini_data.get("manipulated_object", ""))
    task_type = gemini_data.get("task_type", "simple object manip")
    target_object = gemini_data.get("target_prompt", gemini_data.get("target_object", None))

    task_type_norm = str(task_type or "").strip().lower()
    if task_type_norm in {
        "object push",
        "object_push",
        "push",
        "push_object",
        "object pull",
        "object_pull",
        "pull",
        "pull_object",
        "drag",
        "drag_object",
        "put_object_to_object",
        "put_to_object",
        "put",
        "place",
        "place_on_object",
        "place_into_object",
    }:
        task_type_norm = "object push"
    elif task_type_norm in {
        "interactive object manip",
        "interactive_object_manip",
        "interactive",
        "interact",
        "hand interaction",
        "tool interaction",
    }:
        task_type_norm = "interactive object manip"
    else:
        task_type_norm = "simple object manip"
    
    # Build final objects list
    final_objects = []
    
    for sam3d_obj in sam3d_data.get("objects", []):
        object_id = sam3d_obj.get("object_id", 0)
        mesh_path = sam3d_obj.get("mesh_path")
        
        # # Get prompt index from mapping
        # prompt_idx = mask_to_prompt_idx.get(str(object_id), object_id)
        
        # # Get prompt and material from Gemini data
        # if prompt_idx < len(gemini_objects):

        # Get prompt index from mapping. Default to object_id if missing.
        try:
            prompt_idx = int(mask_to_prompt_idx.get(str(object_id), object_id))
        except (TypeError, ValueError):
            prompt_idx = -1  # unresolved → falls through to fallback branch below

        # Get prompt and material from Gemini data
        if 0 <= prompt_idx < len(gemini_objects):
            gemini_obj = gemini_objects[prompt_idx]
            prompt = gemini_obj.get("prompt", "unknown")
            material_type = gemini_obj.get("material_type", "unknown")
        else:
            prompt = f"object_{object_id}"
            material_type = "unknown"
        
        # Get pose (handle null from failed reconstruction)
        pose_data = sam3d_obj.get("pose")
        if pose_data:
            position = pose_data.get("position_m", [0, 0, 0])
            orientation = pose_data.get("orientation_quat_xyzw", [0, 0, 0, 1])
        else:
            position = [0, 0, 0]
            orientation = [0, 0, 0, 1]
        
        # Build object entry with relative mesh path
        if mesh_path:
            mesh_abs = Path(mesh_path)
            if not mesh_abs.is_absolute():
                mesh_rel = mesh_path
            else:
                # Convert to relative path from output_dir
                try:
                    mesh_rel = str(mesh_abs.relative_to(output_dir))
                except ValueError:
                    mesh_rel = str(mesh_path)
        else:
            mesh_rel = ""
        
        obj_entry = {
            "object_id": object_id,
            "prompt": prompt,
            "material_type": material_type,
            "mesh_path": mesh_rel,
            "pose": {
                "position_m": position,
                "orientation_quat_xyzw": orientation
            },
            "is_manipulated": (prompt == manipulated_object),
            "reconstruction_status": sam3d_obj.get("reconstruction_status", "unknown")
        }
        final_objects.append(obj_entry)
    
    # Ensure at least one object is marked as manipulated
    if manipulated_object and not any(o["is_manipulated"] for o in final_objects):
        # Find closest match
        for obj in final_objects:
            if manipulated_object.lower() in obj["prompt"].lower() or obj["prompt"].lower() in manipulated_object.lower():
                obj["is_manipulated"] = True
                break

    # Resolve target_id only when target_object is meaningful.
    prompt_to_object_id = {obj.get("prompt", ""): obj.get("object_id") for obj in final_objects}
    manipulated_id = None
    for obj in final_objects:
        if obj.get("is_manipulated"):
            manipulated_id = obj.get("object_id")
            break

    target_id = None
    target_object_norm = str(target_object or "").strip()
    # if task_type_norm == "object push" and target_object_norm:
    #     target_id = prompt_to_object_id.get(target_object_norm)
    #     if target_id is None:
    #         for obj in final_objects:
    #             p = obj.get("prompt", "")
    #             if target_object_norm.lower() in p.lower() or p.lower() in target_object_norm.lower():
    #                 target_id = obj.get("object_id")
    #                 target_object_norm = p
    #                 break

    #     # If a target was supplied but could not be resolved, ignore it rather than changing the label.
    #     if target_id is None or (manipulated_id is not None and target_id == manipulated_id):
    #         target_id = None
    #         target_object_norm = ""

    if target_object_norm:  # resolve regardless of task_type
    # 1) exact match (case-insensitive)
        for obj in final_objects:
            p = obj.get("prompt", "")
            if p and p.lower() == target_object_norm.lower():
                target_id = obj.get("object_id")
                target_object_norm = p
                break

        # 2) substring fallback — skip empty prompts and the manipulated object
        if target_id is None:
            for obj in final_objects:
                p = obj.get("prompt", "")
                if not p:
                    continue
                if obj.get("object_id") == manipulated_id:
                    continue
                if (target_object_norm.lower() in p.lower()
                        or p.lower() in target_object_norm.lower()):
                    target_id = obj.get("object_id")
                    target_object_norm = p
                    break

        # 3) sanity: target must not equal manipulated
        if manipulated_id is not None and target_id == manipulated_id:
            target_id = None
            target_object_norm = ""

        # 4) if it truly couldn't resolve, drop the label
        if target_id is None:
            target_object_norm = ""
    
    # Build final output
    final_output = {
        "scene_description": gemini_data.get("scene_description", ""),
        "objects": final_objects,
        "manipulated_object": manipulated_object,
        "task_type": task_type_norm,
        "target_object": target_object_norm or None,
        "target_id": target_id,
        "hand_along_object_motion": gemini_data.get("hand_along_object_motion", None),
        "pipeline_metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_objects": len(final_objects),
            "gemini_source": str(gemini_path),
            "sam3d_source": str(sam3d_path)
        }
    }
    
    return final_output


# === Timing instrumentation ===
# Shared state, written to <output_dir>/pipeline_timing.{txt,json} on exit.
_timing = {
    "pipeline_start": None,
    "steps": [],
    "output_dir": None,
    "input_dir": None,
    "finalized": False,
}


def _record_step(name: str, start_time: float, status: str):
    _timing["steps"].append({
        "name": name,
        "duration_s": time.monotonic() - start_time,
        "status": status,
    })


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def _write_timing_report():
    # Idempotent: runs once via atexit so failures still produce a report.
    if _timing["finalized"]:
        return
    if _timing["output_dir"] is None or _timing["pipeline_start"] is None:
        return
    _timing["finalized"] = True

    out_dir = Path(_timing["output_dir"])
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    total = time.monotonic() - _timing["pipeline_start"]
    name_w = max((len(s["name"]) for s in _timing["steps"]), default=10)
    name_w = max(name_w, len("TOTAL"))

    lines = [
        "=" * 70,
        "Pipeline Timing Report",
        "=" * 70,
        f"Input  : {_timing['input_dir']}",
        f"Output : {_timing['output_dir']}",
        f"Stamp  : {datetime.now().isoformat(timespec='seconds')}",
        "-" * 70,
    ]
    for entry in _timing["steps"]:
        dur = _format_duration(entry["duration_s"])
        lines.append(f"  {entry['name']:<{name_w}s}  {dur:>14s}  [{entry['status']}]")
    lines.append("-" * 70)
    lines.append(f"  {'TOTAL':<{name_w}s}  {_format_duration(total):>14s}")
    lines.append("=" * 70)
    text = "\n".join(lines) + "\n"

    try:
        (out_dir / "pipeline_timing.txt").write_text(text)
        json_data = {
            "input_dir": _timing["input_dir"],
            "output_dir": _timing["output_dir"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_s": round(total, 3),
            "steps": [
                {"name": s["name"], "duration_s": round(s["duration_s"], 3), "status": s["status"]}
                for s in _timing["steps"]
            ],
        }
        (out_dir / "pipeline_timing.json").write_text(json.dumps(json_data, indent=2))
    except OSError as e:
        print(f"[WARN] Failed to write timing report: {e}")

    print("\n" + text)


atexit.register(_write_timing_report)


def main():
    parser = argparse.ArgumentParser(
        description="Scene Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YAML config (recommended)
  python run_pipeline.py --config config.yaml
  
  # Using command line arguments
  python run_pipeline.py --input_dir input/ --output_dir output/
  
  # Skip specific steps
  python run_pipeline.py --config config.yaml --skip_gemini
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (alternative to command-line args)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing video.mp4, depth.mp4, cam_K.txt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for all results"
    )
    parser.add_argument(
        "--sam3_env",
        type=str,
        default="sam3",
        help="Conda environment name for SAM3 (default: sam3)"
    )
    parser.add_argument(
        "--sam3d_env",
        type=str,
        default="sam3d-objects",
        help="Conda environment name for SAM3D (default: sam3d-objects)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="auto-pro",
        help="Gemini model for Step 1 (default: auto-pro, e.g., gemini-pro-latest)"
    )
    parser.add_argument(
        "--skip_gemini",
        action="store_true",
        help="Skip Gemini step (use existing gemini_scene.json)"
    )
    parser.add_argument(
        "--skip_sam3",
        action="store_true",
        help="Skip SAM3 step (use existing masks)"
    )
    parser.add_argument(
        "--skip_sam3d",
        action="store_true",
        help="Skip SAM3D step (use existing meshes)"
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip export transforms step"
    )
    parser.add_argument(
        "--skip_visualize",
        action="store_true",
        help="Skip visualization step"
    )
    parser.add_argument(
        "--skip_obj_to_urdf",
        action="store_true",
        help="Skip OBJ to URDF conversion step"
    )
    parser.add_argument(
        "--camera_frame_json",
        type=str,
        help="Path to camera frame JSON (for transformation)"
    )
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        if not HAS_YAML:
            log("PyYAML not installed. Install with: pip install pyyaml", "ERROR")
            sys.exit(1)
        
        config_path = Path(args.config)
        if not config_path.exists():
            log(f"Config file not found: {config_path}", "ERROR")
            sys.exit(1)
        
        log(f"Loading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Override args with config values (command-line args take precedence)
        if not args.input_dir and "input_dir" in config:
            args.input_dir = config["input_dir"]
        if not args.output_dir and "output_dir" in config:
            args.output_dir = config["output_dir"]
        if "sam3_env" in config and args.sam3_env == "sam3":
            args.sam3_env = config["sam3_env"]
        if "sam3d_env" in config and args.sam3d_env == "sam3d-objects":
            args.sam3d_env = config["sam3d_env"]
        if "gemini_model" in config and args.gemini_model == "auto-pro":
            args.gemini_model = config["gemini_model"]
        if "skip_gemini" in config:
            args.skip_gemini = args.skip_gemini or config["skip_gemini"]
        if "skip_sam3" in config:
            args.skip_sam3 = args.skip_sam3 or config["skip_sam3"]
        if "skip_sam3d" in config:
            args.skip_sam3d = args.skip_sam3d or config["skip_sam3d"]
        if "skip_export" in config:
            args.skip_export = args.skip_export or config.get("skip_export", False)
        if "skip_visualize" in config:
            args.skip_visualize = args.skip_visualize or config.get("skip_visualize", False)
        if "skip_obj_to_urdf" in config:
            args.skip_obj_to_urdf = args.skip_obj_to_urdf or config.get("skip_obj_to_urdf", False)
        if not args.camera_frame_json and "camera_frame_json" in config:
            args.camera_frame_json = config["camera_frame_json"]
    
    # Validate required arguments
    if not args.input_dir or not args.output_dir:
        parser.print_help()
        log("\nERROR: --input_dir and --output_dir are required (or use --config)", "ERROR")
        sys.exit(1)
    
    # Resolve paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    scene_dir = Path(args.input_dir) / "scene_capture"
    scripts_dir = Path(__file__).parent / "scripts"
    
    # Validate input directory
    if not input_dir.exists():
        log(f"Input directory does not exist: {input_dir}", "ERROR")
        sys.exit(1)
    
    # Check required files (support both naming conventions)
    video_path = input_dir / "video.mp4"
    if not video_path.exists():
        video_path = input_dir / "color_video.mp4"
    
    # Check for depth - prefer PNG directory over video
    depth_path = input_dir / "depth" / "0.png"
    if not depth_path.exists():
        depth_path = input_dir / "depth.mp4"
        if not depth_path.exists():
            depth_path = input_dir / "depth_video.mp4"
    
    cam_k_path = input_dir / "cam_K.txt"
    if not cam_k_path.exists():
        cam_k_path = input_dir / "cam_params.txt"
    
    # Validate all files exist
    if not video_path.exists():
        log(f"Missing video.mp4 or color_video.mp4 in {input_dir}", "ERROR")
        sys.exit(1)
    if not depth_path.exists():
        log(f"Missing depth.mp4 or depth_video.mp4 in {input_dir}", "ERROR")
        sys.exit(1)
    if not cam_k_path.exists():
        log(f"Missing cam_K.txt or cam_params.txt in {input_dir}", "ERROR")
        sys.exit(1)
    
    # Check GEMINI_API_KEY
    if not args.skip_gemini and not os.environ.get("GEMINI_API_KEY"):
        log("GEMINI_API_KEY environment variable is not set", "ERROR")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize timing instrumentation
    _timing["pipeline_start"] = time.monotonic()
    _timing["output_dir"] = str(output_dir)
    _timing["input_dir"] = str(input_dir)

    log("=" * 60)
    log("SCENE RECONSTRUCTION PIPELINE")
    log("=" * 60)
    log(f"Input:  {input_dir}")
    log(f"Output: {output_dir}")
    log("=" * 60)
    
    # ===== STEP 1: Gemini Scene Analysis =====
    _step_start = time.monotonic()
    if not args.skip_gemini:
        log("STEP 1: Gemini Scene Analysis")
        step1_script = scripts_dir / "step1_gemini.py"

        if not step1_script.exists():
            log(f"Step 1 script not found: {step1_script}", "ERROR")
            _record_step("Step 1: Gemini", _step_start, "failed")
            sys.exit(1)

        # Gemini step runs in current environment (needs google-genai, cv2)
        cmd = [
            sys.executable, str(step1_script),
            "--video", str(video_path),
            "--output_dir", str(output_dir),
            "--model", str(args.gemini_model)
        ]

        success = run_step(cmd, "Gemini Scene Analysis")
        _record_step("Step 1: Gemini", _step_start, "completed" if success else "failed")
        if not success:
            log("Pipeline failed at Step 1: Gemini", "ERROR")
            sys.exit(1)
    else:
        _record_step("Step 1: Gemini", _step_start, "skipped")
        log("STEP 1: Skipped (using existing gemini_scene.json)")
    
    # Verify Gemini output exists
    gemini_output = output_dir / "gemini_scene.json"
    if not gemini_output.exists():
        log(f"Gemini output not found: {gemini_output}", "ERROR")
        sys.exit(1)
    
    # Load Gemini output to get prompts
    with open(gemini_output, 'r') as f:
        gemini_data = json.load(f)
    
    prompts = [obj["prompt"] for obj in gemini_data.get("objects", [])]
    log(f"Found {len(prompts)} objects from Gemini: {prompts}")
    
    # ===== STEP 2: SAM3 Segmentation =====
    _step_start = time.monotonic()
    if not args.skip_sam3:
        log("STEP 2: SAM3 Segmentation")
        step2_script = scripts_dir / "step2_sam3.py"

        if not step2_script.exists():
            log(f"Step 2 script not found: {step2_script}", "ERROR")
            _record_step("Step 2: SAM3", _step_start, "failed")
            sys.exit(1)

        # Pass gemini_scene.json path directly (step2 can read it)
        success = run_conda_step(
            conda_env=args.sam3_env,
            script_path=str(step2_script),
            args=[
                "--image", str(scene_dir / "image" / "0.png"),
                "--output_dir", str(output_dir),
                "--prompts", str(gemini_output)
            ],
            step_name="SAM3 Segmentation"
        )

        _record_step("Step 2: SAM3", _step_start, "completed" if success else "failed")
        if not success:
            log("Pipeline failed at Step 2: SAM3", "ERROR")
            sys.exit(1)
    else:
        _record_step("Step 2: SAM3", _step_start, "skipped")
        log("STEP 2: Skipped (using existing masks)")
    
    # Verify SAM3 output - masks are saved directly in output_dir
    mask_files = list(output_dir.glob("mask_*.png"))
    if not mask_files:
        log(f"No masks found in: {output_dir}", "ERROR")
        sys.exit(1)
    
    mask_count = len(mask_files)
    log(f"Found {mask_count} masks")
    
    # ===== STEP 3: SAM3D Reconstruction =====
    _step_start = time.monotonic()
    if not args.skip_sam3d:
        log("STEP 3: SAM3D Reconstruction")
        step3_script = scripts_dir / "step3_sam3d.py"

        if not step3_script.exists():
            log(f"Step 3 script not found: {step3_script}", "ERROR")
            _record_step("Step 3: SAM3D", _step_start, "failed")
            sys.exit(1)

        success = run_conda_step(
            conda_env=args.sam3d_env,
            script_path=str(step3_script),
            args=[
                "--image", str(scene_dir / "image" / "0.png"),
                "--depth", str(scene_dir / "depth" / "0.png"),
                "--masks", str(output_dir),
                "--cam_k", str(cam_k_path),
                "--output_dir", str(output_dir)
            ],
            step_name="SAM3D Reconstruction"
        )

        _record_step("Step 3: SAM3D", _step_start, "completed" if success else "failed")
        if not success:
            log("Pipeline failed at Step 3: SAM3D", "ERROR")
            sys.exit(1)
    else:
        _record_step("Step 3: SAM3D", _step_start, "skipped")
        log("STEP 3: Skipped (using existing reconstructions)")
    
    # ===== STEP 4: Assemble Final Output =====
    _step_start = time.monotonic()
    log("STEP 4: Assembling Final Output")

    try:
        final_output = assemble_final_output(output_dir)
    except Exception:
        _record_step("Step 4: Assemble Output", _step_start, "failed")
        raise

    # Write final output
    final_output_path = output_dir / "scene_output.json"
    with open(final_output_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    log(f"Final output written to: {final_output_path}")
    _record_step("Step 4: Assemble Output", _step_start, "completed")
    # ===== STEP 5: Export Transformed Scene =====
    _step_start = time.monotonic()
    if not args.skip_export and args.camera_frame_json:
        log("STEP 5: Export Transformed Scene")
        step4_script = scripts_dir / "step4_export_transforms.py"

        if not step4_script.exists():
            log(f"Step 5 script not found: {step4_script}", "ERROR")
            _record_step("Step 5: Export Transforms", _step_start, "failed")
            sys.exit(1)

        camera_frame_path = Path(args.camera_frame_json).resolve()
        if not camera_frame_path.exists():
            log(f"Camera frame JSON not found: {camera_frame_path}", "ERROR")
            _record_step("Step 5: Export Transforms", _step_start, "failed")
            sys.exit(1)

        success = run_step(
            cmd=[
                sys.executable, str(step4_script),
                "--scene_json", str(final_output_path),
                "--camera_frame_json", str(camera_frame_path),
                "--output_json", str(output_dir / "scene_output_new.json")
            ],
            step_name="Export Transformed Scene"
        )

        _record_step("Step 5: Export Transforms", _step_start, "completed" if success else "failed")
        if not success:
            log("Pipeline failed at Step 5: Export Transforms", "ERROR")
            sys.exit(1)
    else:
        _record_step("Step 5: Export Transforms", _step_start, "skipped")
        if args.skip_export:
            log("STEP 5: Skipped (--skip_export)")
        else:
            log("STEP 5: Skipped (no camera_frame_json provided)")
    
    # ===== STEP 6: Visualize Transformed Scene =====
    _step_start = time.monotonic()
    if not args.skip_visualize and args.camera_frame_json and not args.skip_export:
        log("STEP 6: Visualize Transformed Scene")
        step5_script = scripts_dir / "step5_visualize.py"

        if not step5_script.exists():
            log(f"Step 6 script not found: {step5_script}", "ERROR")
            _record_step("Step 6: Visualize", _step_start, "failed")
            sys.exit(1)

        scene_output_new = output_dir / "scene_output_new.json"
        if not scene_output_new.exists():
            log(f"Scene output new not found: {scene_output_new}", "WARNING")
            _record_step("Step 6: Visualize", _step_start, "failed")
        else:
            success = run_step(
                cmd=[
                    sys.executable, str(step5_script),
                    "--scene_json", str(scene_output_new),
                    "--screenshot", str(output_dir / "final_scene_visualization.png")
                ],
                step_name="Visualize Transformed Scene"
            )

            _record_step("Step 6: Visualize", _step_start, "completed" if success else "failed")
            if not success:
                log("Visualization failed, continuing...", "WARNING")
    else:
        _record_step("Step 6: Visualize", _step_start, "skipped")
        log("STEP 6: Skipped")
    
    # ===== STEP 7: OBJ to URDF Conversion =====
    _step_start = time.monotonic()
    if not args.skip_obj_to_urdf and not args.skip_export and args.camera_frame_json:
        log("STEP 7: OBJ to URDF Conversion")
        step6_script = scripts_dir / "step6_obj_to_urdf.py"

        if not step6_script.exists():
            log(f"Step 7 script not found: {step6_script}", "ERROR")
            _record_step("Step 7: OBJ to URDF", _step_start, "failed")
            sys.exit(1)

        scene_output_new = output_dir / "scene_output_new.json"
        gemini_scene = output_dir / "gemini_scene.json"

        if not scene_output_new.exists():
            log(f"Scene output new not found: {scene_output_new}", "ERROR")
            _record_step("Step 7: OBJ to URDF", _step_start, "failed")
            sys.exit(1)

        if not gemini_scene.exists():
            log(f"Gemini scene not found: {gemini_scene}", "ERROR")
            _record_step("Step 7: OBJ to URDF", _step_start, "failed")
            sys.exit(1)

        success = run_step(
            cmd=[
                sys.executable, str(step6_script),
                "--scene_json", str(scene_output_new),
                "--gemini_json", str(gemini_scene),
                "--output_json", str(output_dir / "scene_output_final.json")
            ],
            step_name="OBJ to URDF Conversion"
        )

        _record_step("Step 7: OBJ to URDF", _step_start, "completed" if success else "failed")
        if not success:
            log("Pipeline failed at Step 7: OBJ to URDF", "ERROR")
            sys.exit(1)
    else:
        _record_step("Step 7: OBJ to URDF", _step_start, "skipped")
        if args.skip_obj_to_urdf:
            log("STEP 7: Skipped (--skip_obj_to_urdf)")
        elif args.skip_export:
            log("STEP 7: Skipped (export step was skipped)")
        else:
            log("STEP 7: Skipped (no camera_frame_json provided)")
    
    log("=" * 60)
    log("PIPELINE COMPLETED SUCCESSFULLY")
    log("=" * 60)
    log(f"Results saved to: {output_dir}")
    log("Output files:")
    log(f"  - gemini_scene.json: Scene description from Gemini")
    log(f"  - masks/: Per-object segmentation masks")
    log(f"  - meshes/: Reconstructed 3D meshes (.obj)")
    log(f"  - sam3d_results.json: Mesh paths and poses")
    log(f"  - scene_output.json: Combined output (camera frame)")
    if not args.skip_export and args.camera_frame_json:
        log(f"  - scene_output_new.json: Transformed poses (world frame)")
        log(f"  - transformed_meshes/: Mesh files with local edits")
        log(f"  - final_scene_visualization.png: 3D visualization")
        if not args.skip_obj_to_urdf:
            log(f"  - scene_output_final.json: Final output with URDF paths and physics")
            log(f"  - urdfs/: URDF files with mass, inertia, friction")
    log("=" * 60)


if __name__ == "__main__":
    main()
