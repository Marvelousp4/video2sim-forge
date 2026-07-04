#!/usr/bin/env python3
"""
Step 3: SAM3D 3D Reconstruction

Reconstructs 3D meshes and estimates 6-DOF poses from segmentation masks and depth.
This script MUST be run in the `sam3d-objects` conda environment.

Environment: sam3d-objects (conda activate sam3d-objects)
Required: SAM3D model weights and sam3d_objects package

Input:
    - RGB image (first frame)
    - Depth image (corresponding depth frame, 16-bit PNG in mm)
    - Segmentation masks from Step 2
    - Camera intrinsics (cam_K.txt)
    
Output:
    - obj_0000.obj, obj_0001.obj, ... (3D meshes in meters)
    - sam3d_results.json (poses for each object)
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Set environment before imports
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add SAM3D paths when it is installed from a local checkout.
SAM3D_ROOT = os.environ.get("SAM3D_ROOT", "")
if SAM3D_ROOT:
    sys.path.append(SAM3D_ROOT)
    sys.path.insert(0, os.path.join(SAM3D_ROOT, "notebook"))

try:
    from inference import Inference, load_image
except ImportError as e:
    print(f"ERROR: Cannot import SAM3D. Make sure you're in the 'sam3d-objects' conda environment.")
    print(f"Run: conda activate sam3d-objects")
    print(f"Import error: {e}")
    sys.exit(1)


def load_camera_intrinsics(cam_k_file: str) -> tuple[np.ndarray, dict]:
    """
    Load 3x3 camera intrinsic matrix from file.
    
    Supports two formats:
    1. 3x3 matrix (space or comma separated)
    2. JSON array [fx, fy, cx, cy]
    
    Returns:
        K: 3x3 numpy array
        intrinsics_dict: dict with fx, fy, cx, cy
    """
    content = Path(cam_k_file).read_text().strip()
    
    # Try JSON format first [fx, fy, cx, cy]
    if content.startswith('['):
        import json
        vals = json.loads(content)
        if not (isinstance(vals, list) and len(vals) == 4):
            raise ValueError(f"Expected [fx, fy, cx, cy] in {cam_k_file}, got: {vals}")
        fx, fy, cx, cy = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        # Parse as 3x3 matrix
        K = np.loadtxt(cam_k_file)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    
    return K, {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def depth_to_pointmap(depth_image: np.ndarray, K: np.ndarray, depth_scale: float = 1000.0) -> torch.Tensor:
    """
    Convert depth image to 3D pointmap using camera intrinsics.
    
    Args:
        depth_image: HxW depth map (in millimeters typically)
        K: 3x3 camera intrinsic matrix
        depth_scale: Scale factor to convert depth to meters (1000.0 for mm->m)
    
    Returns:
        pointmap: HxWx3 tensor of 3D points in PyTorch3D camera coordinates
    """
    # Convert depth to meters
    depth = depth_image.astype(np.float32) / depth_scale
    depth[depth <= 0] = np.nan
    
    H, W = depth.shape
    
    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel grid
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    # Unproject to 3D (standard pinhole camera model)
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    
    # Convert to PyTorch3D coordinate system
    # Camera: +X right, +Y down, +Z forward
    # PyTorch3D: +X right, +Y up, +Z forward
    # Transform: negate X and Y
    pointmap = np.stack([-X, -Y, Z], axis=-1)
    
    return torch.tensor(pointmap, dtype=torch.float32)


def run_sam3d_reconstruction(
    image_path: str,
    depth_path: str,
    mask_dir: str,
    cam_k_path: str,
    output_dir: str,
    mesh_ext: str = "obj",
) -> list[dict]:
    """
    Run SAM3D reconstruction for each mask.
    
    Returns:
        List of object dictionaries with mesh_path and pose
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera intrinsics
    print(f"Loading camera intrinsics: {cam_k_path}")
    K, camera_intrinsics = load_camera_intrinsics(cam_k_path)
    print(f"  fx={camera_intrinsics['fx']:.2f}, fy={camera_intrinsics['fy']:.2f}, "
          f"cx={camera_intrinsics['cx']:.2f}, cy={camera_intrinsics['cy']:.2f}")
    
    # Load SAM3D pipeline
    print("\nLoading SAM3D inference pipeline...")
    config_path = os.path.join(SAM3D_ROOT, "checkpoints/hf/pipeline.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SAM3D config not found: {config_path}")
    inference = Inference(config_path, compile=False)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Load depth
    print(f"Loading depth: {depth_path}")
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    
    # Generate pointmap from depth
    print("Generating pointmap from depth...")
    pointmap = depth_to_pointmap(depth, K, depth_scale=1000.0)
    
    # Find all mask files
    mask_paths = sorted(Path(mask_dir).glob("mask_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in: {mask_dir}")
    print(f"Found {len(mask_paths)} masks")
    
    # Load mask-to-prompt mapping if available
    mapping_file = Path(mask_dir) / "mask_to_prompt_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            mask_to_prompt = json.load(f)
        # mask_to_prompt = {int(k): int(v) for k, v in mask_to_prompt.items()}
        # Support both the old format ({"0": 0}) and the new format
        # ({"0": {"prompt_idx": 0, "prompt": "...", "fallback_prompt": "...", "prompt_used": "..."}})
        def _coerce_prompt_idx(v):
            if isinstance(v, dict):
                return int(v["prompt_idx"])
            return int(v)

        mask_to_prompt = {int(k): _coerce_prompt_idx(v) for k, v in mask_to_prompt.items()}
    else:
        mask_to_prompt = {i: i for i in range(len(mask_paths))}
    
    objects = []
    
    for mask_idx, mask_path in enumerate(mask_paths):
        prompt_idx = mask_to_prompt.get(mask_idx, mask_idx)
        print(f"\nProcessing mask {mask_idx} (prompt {prompt_idx}): {mask_path.name}")
        
        # Load mask
        mask = np.array(Image.open(mask_path))
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Check if mask is empty
        if float(mask.sum()) <= 0.0:
            print("  WARNING: Empty mask, skipping reconstruction")
            objects.append({
                "object_id": prompt_idx,
                "mask_file": mask_path.name,
                "mesh_path": None,
                "pose": None,
                "reconstruction_status": "empty_mask",
            })
            continue
        
        try:
            print("  Running SAM3D inference...")
            output = inference(image, mask, seed=42, pointmap=pointmap)
            
            mesh = output["glb"]
            scale = output["scale"][0].cpu().float()
            translation = output["translation"][0].cpu().float()
            rotation_quat = output["rotation"].squeeze().cpu().float()  # wxyz format
            
            # Scale mesh vertices to meters
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            vertices_scaled = vertices * scale
            mesh.vertices = vertices_scaled.cpu().numpy().astype(np.float32)
            
            # Convert quaternion from wxyz to xyzw format
            if rotation_quat.shape[0] == 4:
                rotation_quat_xyzw = torch.tensor([
                    rotation_quat[1],  # x
                    rotation_quat[2],  # y
                    rotation_quat[3],  # z
                    rotation_quat[0],  # w
                ])
            else:
                print(f"  WARNING: Unexpected quaternion shape: {rotation_quat.shape}")
                rotation_quat_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0])
            
            # Save mesh
            mesh_filename = f"obj_{prompt_idx:04d}.{mesh_ext}"
            mesh_path = os.path.join(output_dir, mesh_filename)
            mesh.export(mesh_path)
            print(f"  Saved mesh: {mesh_filename}")
            
            # Store result
            objects.append({
                "object_id": prompt_idx,
                "mask_file": mask_path.name,
                "mesh_path": mesh_filename,
                "pose": {
                    "position_m": translation.tolist(),
                    "orientation_quat_xyzw": rotation_quat_xyzw.tolist(),
                },
                "scale": scale.tolist() if hasattr(scale, 'tolist') else [float(scale)],
                "reconstruction_status": "ok",
            })
            
            print(f"  Position: {translation.tolist()}")
            print(f"  Orientation (xyzw): {rotation_quat_xyzw.tolist()}")
            
        except Exception as e:
            print(f"  ERROR during reconstruction: {e}")
            import traceback
            traceback.print_exc()
            objects.append({
                "object_id": prompt_idx,
                "mask_file": mask_path.name,
                "mesh_path": None,
                "pose": None,
                "reconstruction_status": "error",
                "error": str(e),
            })
    
    # Save results
    results = {
        "camera_intrinsics": camera_intrinsics,
        "objects": objects,
    }
    results_file = os.path.join(output_dir, "sam3d_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_file}")
    
    return objects


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: SAM3D Reconstruction - Generate 3D meshes and poses"
    )
    parser.add_argument("--image", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image (16-bit PNG in mm)")
    parser.add_argument("--masks", required=True, help="Directory containing mask_*.png files")
    parser.add_argument("--cam_k", required=True, help="Path to camera intrinsics file (cam_K.txt)")
    parser.add_argument("--output_dir", required=True, help="Directory to save meshes")
    parser.add_argument("--mesh_ext", default="obj", choices=["obj", "glb"], help="Mesh file format")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STEP 3: SAM3D 3D Reconstruction")
    print("=" * 80)
    print(f"Image: {args.image}")
    print(f"Depth: {args.depth}")
    print(f"Masks: {args.masks}")
    print(f"Intrinsics: {args.cam_k}")
    print(f"Output: {args.output_dir}")
    
    # Run reconstruction
    print("\n" + "-" * 40)
    objects = run_sam3d_reconstruction(
        image_path=args.image,
        depth_path=args.depth,
        mask_dir=args.masks,
        cam_k_path=args.cam_k,
        output_dir=args.output_dir,
        mesh_ext=args.mesh_ext,
    )
    
    # Summary
    reconstructed = sum(1 for o in objects if o.get("mesh_path") is not None)
    print("\n" + "=" * 80)
    print("STEP 3 COMPLETE")
    print("=" * 80)
    print(f"Processed {len(objects)} objects, reconstructed {reconstructed}")


if __name__ == "__main__":
    main()
