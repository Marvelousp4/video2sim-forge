#!/usr/bin/env python3
"""
PyVista-based 3D visualization for scene reconstruction.
Better visualization than matplotlib with proper 3D rendering.
python visualize_pyvista.py --scene_json output/Final_1/scene_output.json --camera_frame_json input/Final_1/camera_frame_pose.json --perm_0_1_2_s_minus1_minus1_1 --rotate_and_drop --align_z_to_table --tag_frame --set_z_zero
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import pyvista as pv


def load_obj_file(obj_path):
    """Load an OBJ file and return vertices, faces, and colors (if available)."""
    vertices = []
    faces = []
    colors = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                # Check for vertex colors (RGB after XYZ)
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


def quat_xyzw_to_matrix(quat_xyzw):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat_xyzw
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def create_coordinate_frame_arrows(scale=0.1, origin=None):
    """Create coordinate frame as three arrows (X=red, Y=green, Z=blue)."""
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])
    
    arrows = []
    colors = ['red', 'green', 'blue']
    directions = [
        [1, 0, 0],  # X
        [0, 1, 0],  # Y
        [0, 0, 1],  # Z
    ]
    
    for i, (direction, color) in enumerate(zip(directions, colors)):
        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            scale=scale,
            tip_length=0.3,
            tip_radius=0.1,
            shaft_radius=0.03,
        )
        arrows.append((arrow, color))
    
    return arrows


def create_transformed_frame_arrows(T, scale=0.1):
    """Create coordinate frame arrows at a specific transformation matrix pose."""
    origin = T[:3, 3]
    R = T[:3, :3]
    
    arrows = []
    colors = ['red', 'green', 'blue']
    
    for i, color in enumerate(colors):
        local_dir = np.array([0.0, 0.0, 0.0])
        local_dir[i] = 1.0
        world_dir = R @ local_dir
        
        arrow = pv.Arrow(
            start=origin,
            direction=world_dir,
            scale=scale,
            tip_length=0.3,
            tip_radius=0.1,
            shaft_radius=0.03,
        )
        arrows.append((arrow, color))
    
    return arrows


def transform_vertices(vertices, T):
    """Transform vertices using a 4x4 transformation matrix."""
    vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    transformed = (T @ vertices_homogeneous.T).T
    return transformed[:, :3]


def apply_perm_0_1_2_s_minus1_minus1_1(scene_data):
    """
    Step 1: Apply perm(0,1,2)_s-1-11 transformation.
    S = diag([-1, -1, 1])
    - Position: p' = S @ p = [-x, -y, z]
    - Rotation: R' = S @ R (pre-multiplication, NOT conjugation)
    """
    S = np.diag([-1, -1, 1])
    
    print("\n" + "="*60)
    print("Step 1: perm(0,1,2)_s-1-11 transformation")
    print("S = diag([-1, -1, 1]) → Negate X and Y, keep Z")
    print("="*60)
    
    transformed_scene = {'objects': [], 'scene_description': scene_data.get('scene_description', '')}
    
    for idx, obj_data in enumerate(scene_data['objects']):
        prompt = obj_data['prompt']
        position = np.array(obj_data['pose']['position_m'])
        quat_xyzw = obj_data['pose']['orientation_quat_xyzw']
        
        # Transform position: p' = S @ p
        new_position = S @ position
        
        # Transform rotation: R' = S @ R (pre-multiply, not conjugate)
        R = quat_xyzw_to_matrix(quat_xyzw)
        new_R = S @ R
        new_quat_xyzw = Rotation.from_matrix(new_R).as_quat()  # Returns [x, y, z, w]
        
        print(f"\nObject {idx}: {prompt}")
        print(f"  Old position: {position}")
        print(f"  New position: {new_position}")
        
        # Create new object data
        new_obj = obj_data.copy()
        new_obj['pose'] = {
            'position_m': new_position.tolist(),
            'orientation_quat_xyzw': new_quat_xyzw.tolist()
        }
        transformed_scene['objects'].append(new_obj)
    
    print("\n" + "="*60)
    return transformed_scene


def rotate_and_drop_in_local_frame(scene_data):
    """
    Step 2: Edit MESH in local frame:
    - Rotate 90° around X
    - Rotate 180° around Z (flip)
    - Drop to bottom
    DO NOT change pose - only modify vertices in object's local coordinate system.
    """
    print("\n" + "="*60)
    print("Step 2: Edit MESH in local frame (90°X + 180°Z + drop), pose unchanged")
    print("="*60)
    
    transformed_scene = {'objects': [], 'scene_description': scene_data.get('scene_description', '')}
    
    # Local adjustment: rotate 90° around X, then 180° around Z
    R_x90 = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    R_z180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    R_local_adjust = R_z180 @ R_x90  # Apply X first, then Z
    
    for idx, obj_data in enumerate(scene_data['objects']):
        mesh_path = obj_data['mesh_path']
        prompt = obj_data['prompt']
        
        # Skip objects without mesh
        if not mesh_path or mesh_path == "":
            print(f"\nObject {idx}: {prompt} - SKIPPED (no mesh)")
            transformed_scene['objects'].append(obj_data.copy())
            continue
        
        # Load original mesh
        vertices, faces, colors = load_obj_file(mesh_path)
        
        print(f"\nObject {idx}: {prompt}")
        print(f"  Original mesh Z range: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
        
        # Rotate vertices by 90° X then 180° Z in local frame
        rotated_vertices = (R_local_adjust @ vertices.T).T
        z_min_rotated = rotated_vertices[:, 2].min()
        
        # Shift vertices so bottom is at Z=0 in local frame
        adjusted_vertices = rotated_vertices.copy()
        adjusted_vertices[:, 2] += (-z_min_rotated)
        
        print(f"  After 90°X + 180°Z rotation, Z range: [{z_min_rotated:.4f}, {rotated_vertices[:, 2].max():.4f}]")
        print(f"  Shifted by {-z_min_rotated:.4f}m to place bottom at local Z=0")
        print(f"  POSE NOT CHANGED (frame unchanged)")
        
        # Create new object data - KEY: pose is NOT changed!
        new_obj = obj_data.copy()
        new_obj['_transformed_vertices'] = adjusted_vertices
        if colors is not None:
            new_obj['_vertex_colors'] = colors  # Store colors
        transformed_scene['objects'].append(new_obj)
    
    print("\n" + "="*60)
    return transformed_scene


def align_z_axis_to_table(scene_data, camera_frame_json):
    """
    Step 2.5: Align each object's Z-axis perpendicular to table (in camera frame).
    Changes pose rotation but NOT position.
    Makes object Z-axis point same direction as table normal (tag 0's Z-axis).
    """
    # Load camera frame data
    with open(camera_frame_json, 'r') as f:
        camera_tags = json.load(f)
    
    # Find reference tag (tag 0) - its Z-axis is table normal
    ref_tag = None
    for tag in camera_tags:
        if tag['tag_index'] == 0:
            ref_tag = tag
            break
    
    if ref_tag is None:
        print("Warning: Tag 0 not found, skipping Z-axis alignment")
        return scene_data
    
    # Get table normal direction (tag 0's Z-axis in camera frame)
    ref_euler_deg = np.array(ref_tag['orientation_deg_XYZ(deg)'])
    ref_rotation = Rotation.from_euler('XYZ', ref_euler_deg, degrees=True).as_matrix()
    table_normal_in_cam = ref_rotation[:, 2]  # Z-axis column
    
    print("\n" + "="*60)
    print("Step 2.5: Align object Z-axes perpendicular to table")
    print(f"Table normal (tag 0 Z-axis) in camera: {table_normal_in_cam}")
    print("="*60)
    
    transformed_scene = {'objects': [], 'scene_description': scene_data.get('scene_description', '')}
    
    for idx, obj_data in enumerate(scene_data['objects']):
        prompt = obj_data['prompt']
        position = np.array(obj_data['pose']['position_m'])
        quat_xyzw = obj_data['pose']['orientation_quat_xyzw']
        
        # Get current object rotation
        R_old = quat_xyzw_to_matrix(quat_xyzw)
        old_z_axis = R_old[:, 2]
        
        # Create new rotation that:
        # - Z-axis points along table normal
        # - Keep X, Y axes reasonable (minimize rotation)
        new_z = table_normal_in_cam / np.linalg.norm(table_normal_in_cam)
        
        # Choose arbitrary X axis perpendicular to Z
        # Use old X axis as reference, project it onto plane perpendicular to new Z
        old_x = R_old[:, 0]
        new_x = old_x - np.dot(old_x, new_z) * new_z
        if np.linalg.norm(new_x) < 0.1:
            # If old X was parallel to new Z, use old Y instead
            old_y = R_old[:, 1]
            new_x = old_y - np.dot(old_y, new_z) * new_z
        new_x = new_x / np.linalg.norm(new_x)
        
        # Y axis = Z cross X (right-hand rule)
        new_y = np.cross(new_z, new_x)
        new_y = new_y / np.linalg.norm(new_y)
        
        # Build new rotation matrix
        R_new = np.column_stack([new_x, new_y, new_z])
        new_quat_xyzw = Rotation.from_matrix(R_new).as_quat()
        
        print(f"\nObject {idx}: {prompt}")
        print(f"  Old Z-axis: {old_z_axis}")
        print(f"  New Z-axis: {new_z}")
        print(f"  Position unchanged: {position}")
        
        # Create new object data with aligned rotation
        new_obj = obj_data.copy()
        new_obj['pose'] = {
            'position_m': position.tolist(),
            'orientation_quat_xyzw': new_quat_xyzw.tolist()
        }
        # Keep transformed vertices if they exist
        if '_transformed_vertices' in obj_data:
            new_obj['_transformed_vertices'] = obj_data['_transformed_vertices']
        
        transformed_scene['objects'].append(new_obj)
    
    print("\n" + "="*60)
    return transformed_scene


def transform_to_tag_frame(scene_data, camera_frame_json, set_z_to_zero=False):
    """
    Step 3: Transform all objects from camera frame to tag 0 (world/table) frame.
    Uses T_world_obj = T_world_cam @ T_cam_obj
    
    Args:
        set_z_to_zero: If True, set all objects' Z position to 0 (place on table surface)
    """
    # Load camera frame data
    with open(camera_frame_json, 'r') as f:
        camera_tags = json.load(f)
    
    # Find reference tag (tag 0)
    ref_tag = None
    for tag in camera_tags:
        if tag['tag_index'] == 0:
            ref_tag = tag
            break
    
    if ref_tag is None:
        print("Warning: Tag 0 not found in camera frame JSON")
        return scene_data
    
    # Get T_cam_tag0
    ref_position = np.array(ref_tag['position(m)'])
    ref_euler_deg = np.array(ref_tag['orientation_deg_XYZ(deg)'])
    ref_rotation = Rotation.from_euler('XYZ', ref_euler_deg, degrees=True).as_matrix()
    
    T_cam_tag0 = np.eye(4)
    T_cam_tag0[:3, :3] = ref_rotation
    T_cam_tag0[:3, 3] = ref_position
    
    # Invert to get T_tag0_cam (world to camera)
    T_world_cam = np.linalg.inv(T_cam_tag0)
    
    print("\n" + "="*60)
    print("Step 3: Transform to Tag 0 (World/Table) Frame")
    if set_z_to_zero:
        print("  + Setting all Z positions to 0 (place on table surface)")
    print("="*60)
    print(f"\nTag 0 in camera frame:")
    print(f"  Position: {ref_position}")
    print(f"  Rotation: {ref_euler_deg}")
    print(f"\nCamera in world frame:")
    print(f"  Position: {T_world_cam[:3, 3]}")
    
    transformed_scene = {'objects': [], 'scene_description': scene_data.get('scene_description', '')}
    
    for idx, obj_data in enumerate(scene_data['objects']):
        prompt = obj_data['prompt']
        position_cam = np.array(obj_data['pose']['position_m'])
        quat_xyzw_cam = obj_data['pose']['orientation_quat_xyzw']
        
        # Get T_cam_obj
        R_cam_obj = quat_xyzw_to_matrix(quat_xyzw_cam)
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = position_cam
        
        # Transform: T_world_obj = T_world_cam @ T_cam_obj
        T_world_obj = T_world_cam @ T_cam_obj
        
        new_position = T_world_obj[:3, 3]
        
        # Set Z to 0 if requested (place on table)
        if set_z_to_zero:
            new_position[2] = 0.0
        
        new_R = T_world_obj[:3, :3]
        new_quat_xyzw = Rotation.from_matrix(new_R).as_quat()
        
        print(f"\nObject {idx}: {prompt}")
        print(f"  Camera frame position: {position_cam}")
        print(f"  World frame position: {new_position}")
        
        # Create new object data
        new_obj = obj_data.copy()
        new_obj['pose'] = {
            'position_m': new_position.tolist(),
            'orientation_quat_xyzw': new_quat_xyzw.tolist()
        }
        if '_transformed_vertices' in obj_data:
            new_obj['_transformed_vertices'] = obj_data['_transformed_vertices']
        
        transformed_scene['objects'].append(new_obj)
    
    print("\n" + "="*60)
    return transformed_scene, T_world_cam


def main():
    parser = argparse.ArgumentParser(description='PyVista 3D Visualization for Scene Reconstruction')
    parser.add_argument('--scene_json', type=str, required=True, help='Path to scene_output.json')
    parser.add_argument('--camera_frame_json', type=str, default=None, help='Path to camera frame JSON (for tag poses)')
    parser.add_argument('--perm_0_1_2_s_minus1_minus1_1', action='store_true', 
                       help='Apply perm(0,1,2)_s-1-11 transformation (negate X and Y)')
    parser.add_argument('--rotate_and_drop', action='store_true',
                       help='Step 1: Rotate 90° X in local frame and move center to bottom')    
    parser.add_argument('--align_z_to_table', action='store_true',
                       help='Step 2.5: Align object Z-axis perpendicular to table (change rotation, not position)')    
    parser.add_argument('--tag_frame', action='store_true',
                       help='Step 2: Transform to tag 0 (world/table) frame')    
    parser.add_argument('--set_z_zero', action='store_true',
                       help='In tag frame: Set all objects Z position to 0 (place on table surface)')    
    args = parser.parse_args()
    
    # Load scene data
    scene_json_path = Path(args.scene_json)
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    
    # Fix mesh paths to be relative to scene_json directory
    scene_dir = scene_json_path.parent
    for obj_data in scene_data['objects']:
        if 'mesh_path' in obj_data and obj_data['mesh_path']:
            mesh_path = Path(obj_data['mesh_path'])
            # If absolute path doesn't exist, try relative to scene_json directory
            if not mesh_path.exists():
                mesh_filename = mesh_path.name
                local_mesh_path = scene_dir / mesh_filename
                if local_mesh_path.exists():
                    obj_data['mesh_path'] = str(local_mesh_path)
    
    # Apply transformation if requested
    if args.perm_0_1_2_s_minus1_minus1_1:
        scene_data = apply_perm_0_1_2_s_minus1_minus1_1(scene_data)
    
    # Step 1: Rotate and drop in local frame
    use_transformed_mesh = False
    if args.rotate_and_drop:
        scene_data = rotate_and_drop_in_local_frame(scene_data)
        use_transformed_mesh = True
    
    # Step 2.5: Align Z-axis to table normal (change rotation, not position)
    if args.align_z_to_table and args.camera_frame_json:
        scene_data = align_z_axis_to_table(scene_data, args.camera_frame_json)
    
    # Step 3: Transform to tag frame
    T_world_cam = None
    frame_name = "Camera Frame"
    if args.tag_frame and args.camera_frame_json:
        scene_data, T_world_cam = transform_to_tag_frame(scene_data, args.camera_frame_json, 
                                                          set_z_to_zero=args.set_z_zero)
        frame_name = "World/Table Frame (Tag 0)"
    
    print(f"\nScene Description: {scene_data.get('scene_description', '')}")
    print(f"Number of objects: {len(scene_data['objects'])}")
    
    # Create PyVista plotter
    pv.set_plot_theme('document')
    plotter = pv.Plotter(window_size=[1600, 1000])
    plotter.add_title(f"Scene Visualization in {frame_name}", font_size=16)
    
    # Add camera frame at origin (large, labeled)
    frame_arrows = create_coordinate_frame_arrows(scale=0.3)
    for arrow, color in frame_arrows:
        plotter.add_mesh(arrow, color=color)
    
    # Determine which frames to show
    is_world_frame = args.tag_frame and args.camera_frame_json
    
    if is_world_frame:
        # We're in world frame - show tag 0 at origin, camera elsewhere
        plotter.add_point_labels(
            [[0.35, 0, 0]], 
            ['TABLE/WORLD\nFRAME (TAG 0)'], 
            font_size=20, 
            text_color='darkgreen',
            always_visible=True,
            shape_opacity=0.7,
            shape='rounded_rect',
            fill_shape=True,
            shape_color='yellow'
        )
        
        # Show camera frame if T_world_cam available
        if T_world_cam is not None:
            cam_arrows = create_transformed_frame_arrows(T_world_cam, scale=0.25)
            for arrow, color in cam_arrows:
                plotter.add_mesh(arrow, color=color, opacity=0.8)
            
            cam_pos = T_world_cam[:3, 3]
            label_pos = cam_pos + np.array([0.15, 0.1, 0])
            plotter.add_point_labels(
                [label_pos], 
                ['CAMERA\nFRAME'], 
                font_size=18, 
                text_color='blue',
                always_visible=True,
                shape_opacity=0.8,
                shape='rounded_rect',
                fill_shape=True,
                shape_color='lightblue'
            )
            
            # Draw table plane at Z=0
            table_size = 0.8
            xx, yy = np.meshgrid(np.linspace(-table_size, table_size, 10),
                                 np.linspace(-table_size, table_size, 10))
            zz = np.zeros_like(xx)
            table_plane = pv.StructuredGrid(xx, yy, zz)
            plotter.add_mesh(table_plane, opacity=0.15, color='gray', show_edges=True)
    
    else:
        # We're in camera frame - show camera at origin, tag 0 elsewhere
        plotter.add_point_labels(
            [[0.35, 0, 0]], 
            ['CAMERA FRAME'], 
            font_size=20, 
            text_color='blue',
            always_visible=True,
            shape_opacity=0.7,
            shape='rounded_rect',
            fill_shape=True,
            shape_color='lightblue'
        )
    
    # Load and show Tag 0 frame if camera_frame_json provided and in camera frame
    if args.camera_frame_json and not is_world_frame:
        with open(args.camera_frame_json, 'r') as f:
            camera_tags = json.load(f)
        
        ref_tag = None
        for tag in camera_tags:
            if tag['tag_index'] == 0:
                ref_tag = tag
                break
        
        if ref_tag is not None:
            ref_position = np.array(ref_tag['position(m)'])
            ref_euler_deg = np.array(ref_tag['orientation_deg_XYZ(deg)'])
            ref_rotation = Rotation.from_euler('XYZ', ref_euler_deg, degrees=True).as_matrix()
            
            T_cam_tag0 = np.eye(4)
            T_cam_tag0[:3, :3] = ref_rotation
            T_cam_tag0[:3, 3] = ref_position
            
            print(f"\n=== Tag 0 (Table Frame) in Camera Coordinates ===")
            print(f"Position: {ref_position}")
            print(f"Euler XYZ (deg): {ref_euler_deg}")
            
            # Add tag 0 frame arrows
            tag_arrows = create_transformed_frame_arrows(T_cam_tag0, scale=0.25)
            for arrow, color in tag_arrows:
                plotter.add_mesh(arrow, color=color, opacity=0.8)
            
            # Add label for tag 0 frame
            label_pos = ref_position + np.array([0.15, 0.1, 0])
            plotter.add_point_labels(
                [label_pos], 
                ['TAG 0\n(TABLE)'], 
                font_size=18, 
                text_color='darkgreen',
                always_visible=True,
                shape_opacity=0.8,
                shape='rounded_rect',
                fill_shape=True,
                shape_color='yellow'
            )
    
    # Color palette for objects
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'yellow', 'magenta', 'cyan', 'orange']
    
    # Process each object
    print(f"\n=== Objects in {frame_name} ===")
    for idx, obj_data in enumerate(scene_data['objects']):
        mesh_path = obj_data['mesh_path']
        prompt = obj_data['prompt']
        position = np.array(obj_data['pose']['position_m'])
        quat_xyzw = obj_data['pose']['orientation_quat_xyzw']
        
        # Skip objects without mesh
        if not mesh_path or mesh_path == "":
            print(f"\nObject {idx}: {prompt} - SKIPPED (no mesh)")
            continue
        
        print(f"\nObject {idx}: {prompt}")
        print(f"  Position: {position}")
        
        # Load mesh - use transformed vertices if available
        if use_transformed_mesh and '_transformed_vertices' in obj_data:
            vertices = obj_data['_transformed_vertices']
            _, faces, colors = load_obj_file(mesh_path)  # Still need faces and colors
            if '_vertex_colors' in obj_data:
                colors = obj_data['_vertex_colors']
            print(f"  Using transformed mesh (rotated & dropped in local frame)")
        else:
            vertices, faces, colors = load_obj_file(mesh_path)
        
        # Create transformation matrix
        R = quat_xyzw_to_matrix(quat_xyzw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        # Transform vertices
        transformed_vertices = transform_vertices(vertices, T)
        
        # Print Z-range
        z_min = transformed_vertices[:, 2].min()
        z_max = transformed_vertices[:, 2].max()
        print(f"  Z-range: [{z_min:.4f}, {z_max:.4f}]")
        
        # Create PyVista mesh - use vertex colors if available
        mesh = pv.PolyData(transformed_vertices)
        
        if colors is not None:
            # Use vertex colors (convert 0-1 float to 0-255 uint8)
            rgb_colors = (colors * 255).astype(np.uint8)
            mesh['RGB'] = rgb_colors
            plotter.add_mesh(mesh, scalars='RGB', rgb=True, point_size=5,
                           render_points_as_spheres=True, opacity=1.0,
                           label=f'{idx}: {prompt[:30]}')
        else:
            # Fallback to solid color
            fallback_colors = ['lightcoral', 'lightgreen', 'lightblue', 'yellow', 'magenta', 'cyan', 'orange']
            if len(faces) > 0:
                # Convert faces to PyVista format
                pv_faces = []
                for face in faces:
                    pv_faces.append(len(face))
                    pv_faces.extend(face)
                
                mesh = pv.PolyData(transformed_vertices, pv_faces)
                plotter.add_mesh(mesh, color=fallback_colors[idx % len(fallback_colors)], 
                               opacity=0.6, label=f'{idx}: {prompt[:30]}...')
            else:
                # Just show as point cloud
                plotter.add_mesh(mesh, color=fallback_colors[idx % len(fallback_colors)], 
                               point_size=3, render_points_as_spheres=True,
                               label=f'{idx}: {prompt[:30]}...')
        
        # Add small object frame
        obj_arrows = create_transformed_frame_arrows(T, scale=0.05)
        for arrow, color in obj_arrows:
            plotter.add_mesh(arrow, color=color, opacity=0.8)
    
    # Set up the view
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=3)
    plotter.add_legend(bcolor='white', face='circle', size=(0.2, 0.3))
    plotter.camera_position = 'iso'
    
    # Add grid for reference
    plotter.show_grid()
    
    print("\n" + "="*60)
    print("Visualization launched! Use mouse to rotate/zoom.")
    print("="*60)
    
    plotter.show()


if __name__ == '__main__':
    main()
