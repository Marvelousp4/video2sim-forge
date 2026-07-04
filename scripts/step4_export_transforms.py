#!/usr/bin/env python3
"""
Export transformed scene with:
1. Per-object local mesh edits (rotate_and_drop) saved as new OBJ files
2. Global pose transformations (align_z, tag_frame, set_z_zero) saved in scene_output_new.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_obj_file(obj_path):
    """Load an OBJ file and return vertices, faces, and colors."""
    vertices = []
    faces = []
    colors = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face_vertices = []
                for part in parts:
                    vertex_idx = int(part.split('/')[0]) - 1
                    face_vertices.append(vertex_idx)
                if len(face_vertices) >= 3:
                    faces.append(face_vertices)
    
    colors_array = np.array(colors) if colors else None
    return np.array(vertices), faces, colors_array


def save_obj_file(obj_path, vertices, faces, colors=None):
    """Save an OBJ file with vertices, faces, and optional colors."""
    with open(obj_path, 'w') as f:
        # Write vertices
        for i, v in enumerate(vertices):
            if colors is not None and i < len(colors):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {colors[i][0]:.6f} {colors[i][1]:.6f} {colors[i][2]:.6f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {' '.join([str(idx+1) for idx in face])}\n")


def quat_xyzw_to_matrix(quat_xyzw):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat_xyzw
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def apply_perm_transformation(position, quat_xyzw):
    """Apply perm(0,1,2)_s-1-11: negate X and Y."""
    S = np.diag([-1, -1, 1])
    
    # Transform position
    new_position = S @ position
    
    # Transform orientation: R' = S @ R @ S.T
    R = quat_xyzw_to_matrix(quat_xyzw)
    R_new = S @ R @ S.T
    new_quat_xyzw = Rotation.from_matrix(R_new).as_quat()
    
    return new_position, new_quat_xyzw


def apply_rotate_and_drop_to_mesh(vertices, faces, colors):
    """
    Apply local-frame mesh edits:
    - Rotate 90° around X
    - Rotate 180° around Z
    - Drop to bottom (min Z = 0)
    """
    # Local adjustment: rotate 90° around X, then 180° around Z
    R_x90 = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    R_z180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    R_local_adjust = R_z180 @ R_x90
    
    # Rotate vertices
    rotated_vertices = (R_local_adjust @ vertices.T).T
    
    # Drop to bottom
    z_min = rotated_vertices[:, 2].min()
    adjusted_vertices = rotated_vertices.copy()
    adjusted_vertices[:, 2] -= z_min
    
    return adjusted_vertices, faces, colors


def align_z_to_table(position, quat_xyzw, camera_frame_json):
    """
    Align object Z-axis perpendicular to table.
    Returns new quaternion, position unchanged.
    """
    # Load camera frame data
    with open(camera_frame_json, 'r') as f:
        camera_tags = json.load(f)
    
    # Find tag 0
    ref_tag = None
    for tag in camera_tags:
        if tag['tag_index'] == 0:
            ref_tag = tag
            break
    
    if ref_tag is None:
        return quat_xyzw  # No change if tag 0 not found
    
    # Get table normal (tag 0's Z-axis in camera frame)
    ref_euler_deg = np.array(ref_tag['orientation_deg_XYZ(deg)'])
    ref_rotation = Rotation.from_euler('XYZ', ref_euler_deg, degrees=True).as_matrix()
    table_normal_in_cam = ref_rotation[:, 2]
    
    # Get current object rotation
    R_old = quat_xyzw_to_matrix(quat_xyzw)
    
    # Create new rotation with Z-axis aligned to table normal
    new_z = table_normal_in_cam / np.linalg.norm(table_normal_in_cam)
    
    # Project old X onto plane perpendicular to new Z
    old_x = R_old[:, 0]
    new_x = old_x - np.dot(old_x, new_z) * new_z
    if np.linalg.norm(new_x) < 0.1:
        old_y = R_old[:, 1]
        new_x = old_y - np.dot(old_y, new_z) * new_z
    new_x = new_x / np.linalg.norm(new_x)
    
    # Y = Z cross X
    new_y = np.cross(new_z, new_x)
    new_y = new_y / np.linalg.norm(new_y)
    
    # Build new rotation matrix
    R_new = np.column_stack([new_x, new_y, new_z])
    new_quat_xyzw = Rotation.from_matrix(R_new).as_quat()
    
    return new_quat_xyzw


def transform_to_tag_frame(position, quat_xyzw, camera_frame_json, set_z_zero=False):
    """
    Transform pose from camera frame to world/table frame (tag 0).
    Returns new position and quaternion.
    """
    # Load camera frame data
    with open(camera_frame_json, 'r') as f:
        camera_tags = json.load(f)
    
    # Find tag 0
    ref_tag = None
    for tag in camera_tags:
        if tag['tag_index'] == 0:
            ref_tag = tag
            break
    
    if ref_tag is None:
        return position, quat_xyzw  # No change if tag 0 not found
    
    # Build T_cam_tag0
    ref_position = np.array(ref_tag['position(m)'])
    ref_euler_deg = np.array(ref_tag['orientation_deg_XYZ(deg)'])
    ref_rotation = Rotation.from_euler('XYZ', ref_euler_deg, degrees=True).as_matrix()
    
    T_cam_tag0 = np.eye(4)
    T_cam_tag0[:3, :3] = ref_rotation
    T_cam_tag0[:3, 3] = ref_position
    
    # Invert to get T_world_cam (world = tag0)
    T_world_cam = np.linalg.inv(T_cam_tag0)
    
    # Build T_cam_obj
    R_cam_obj = quat_xyzw_to_matrix(quat_xyzw)
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_cam_obj
    T_cam_obj[:3, 3] = position
    
    # Transform: T_world_obj = T_world_cam @ T_cam_obj
    T_world_obj = T_world_cam @ T_cam_obj
    
    new_position = T_world_obj[:3, 3]
    
    # Set Z to 0 if requested
    if set_z_zero:
        new_position[2] = 0.0
    
    new_R = T_world_obj[:3, :3]
    new_quat_xyzw = Rotation.from_matrix(new_R).as_quat()
    
    return new_position, new_quat_xyzw


def main():
    parser = argparse.ArgumentParser(description='Export transformed scene with new OBJ files and poses')
    parser.add_argument('--scene_json', type=str, required=True, help='Path to input scene_output.json')
    parser.add_argument('--camera_frame_json', type=str, required=True, help='Path to camera frame JSON')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output scene_output_new.json')
    parser.add_argument('--output_mesh_dir', type=str, help='Directory for transformed meshes (default: same as output_json)')
    
    args = parser.parse_args()
    
    # Setup paths
    scene_json_path = Path(args.scene_json)
    output_json_path = Path(args.output_json)
    
    if args.output_mesh_dir:
        output_mesh_dir = Path(args.output_mesh_dir)
    else:
        output_mesh_dir = output_json_path.parent / "transformed_meshes"
    
    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scene data
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    
    # Fix mesh paths to be relative to scene_json directory
    scene_dir = scene_json_path.parent
    for obj_data in scene_data['objects']:
        if 'mesh_path' in obj_data and obj_data['mesh_path']:
            mesh_path = Path(obj_data['mesh_path'])
            if not mesh_path.exists():
                mesh_filename = mesh_path.name
                local_mesh_path = scene_dir / mesh_filename
                if local_mesh_path.exists():
                    obj_data['mesh_path'] = str(local_mesh_path)
    
    print("=" * 60)
    print("EXPORT TRANSFORMED SCENE")
    print("=" * 60)
    print(f"Input scene:  {scene_json_path}")
    print(f"Camera frame: {args.camera_frame_json}")
    print(f"Output JSON:  {output_json_path}")
    print(f"Output meshes: {output_mesh_dir}")
    print("=" * 60)
    
    new_scene = scene_data.copy()
    new_scene['objects'] = []
    
    for idx, obj_data in enumerate(scene_data['objects']):
        prompt = obj_data['prompt']
        mesh_path = obj_data.get('mesh_path', '')
        
        print(f"\nObject {idx}: {prompt}")
        
        # Skip objects without mesh
        if not mesh_path or mesh_path == "":
            print(f"  SKIPPED (no mesh)")
            new_obj = obj_data.copy()
            new_scene['objects'].append(new_obj)
            continue
        
        # Get original pose
        position = np.array(obj_data['pose']['position_m'])
        quat_xyzw = np.array(obj_data['pose']['orientation_quat_xyzw'])
        
        print(f"  Original position: {position}")
        print(f"  Original orientation: {quat_xyzw}")
        
        # Step 1: Apply perm transformation to pose
        position_perm, quat_perm = apply_perm_transformation(position, quat_xyzw)
        print(f"  After perm: position={position_perm}")
        
        # Step 2: Apply rotate_and_drop to MESH (save new OBJ)
        vertices, faces, colors = load_obj_file(mesh_path)
        print(f"  Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        vertices_transformed, faces_transformed, colors_transformed = \
            apply_rotate_and_drop_to_mesh(vertices, faces, colors)
        
        # Save transformed mesh
        mesh_filename = Path(mesh_path).stem
        new_mesh_path = output_mesh_dir / f"{mesh_filename}_transformed.obj"
        save_obj_file(new_mesh_path, vertices_transformed, faces_transformed, colors_transformed)
        print(f"  Saved transformed mesh: {new_mesh_path}")
        
        # Step 3: Apply align_z_to_table to POSE
        quat_aligned = align_z_to_table(position_perm, quat_perm, args.camera_frame_json)
        print(f"  After align_z: orientation={quat_aligned}")
        
        # Step 4: Apply tag_frame transformation to POSE
        position_world, quat_world = transform_to_tag_frame(
            position_perm, quat_aligned, args.camera_frame_json, set_z_zero=True
        )
        print(f"  After tag_frame+set_z_zero: position={position_world}, orientation={quat_world}")
        
        # Create new object data with relative mesh path
        new_mesh_rel = new_mesh_path.relative_to(output_json_path.parent)
        
        new_obj = obj_data.copy()
        new_obj['mesh_path'] = str(new_mesh_rel)
        new_obj['pose'] = {
            'position_m': position_world.tolist(),
            'orientation_quat_xyzw': quat_world.tolist()
        }
        
        new_scene['objects'].append(new_obj)
    
    # Save new scene JSON
    with open(output_json_path, 'w') as f:
        json.dump(new_scene, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"New scene JSON: {output_json_path}")
    print(f"Transformed meshes: {output_mesh_dir}")
    print(f"Total objects: {len(new_scene['objects'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
