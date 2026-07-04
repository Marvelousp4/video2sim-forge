#!/usr/bin/env python3
"""
Pure visualization script for transformed scene.
Loads transformed meshes and scene_output_new.json and displays them in the world/tag frame.
All transformations are already baked into the meshes and poses.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pyvista as pv
import argparse

def load_real_camera_view(
    camera_frame_json_path: str,
    tag_index: int = 0,
    distance_scale: float = 1.0,
    focal_point: tuple[float, float, float] | None = None,
):
    """
    distance_scale: multiplies the camera-to-focal distance. >1 pulls the camera
        back along its viewing axis. 2.0 doubles the distance.
    focal_point: if given (x,y,z in world frame), the camera looks at this point
        instead of '1 m along the optical axis'. Useful for centering on the
        table origin.
    """
    with open(camera_frame_json_path, "r") as f:
        cam_poses = json.load(f)

    ref = next((p for p in cam_poses if p.get("tag_index") == tag_index), None)
    if ref is None:
        raise ValueError(
            f"No camera pose entry with tag_index={tag_index} in {camera_frame_json_path}"
        )

    # Tag pose expressed in the camera frame.
    T_cam_from_tag = np.eye(4)
    T_cam_from_tag[:3, :3] = Rotation.from_euler(
        "XYZ", ref["orientation_deg_XYZ(deg)"], degrees=True
    ).as_matrix()
    T_cam_from_tag[:3, 3] = ref["position(m)"]

    # Invert -> camera pose in tag/world frame.
    T_world_from_cam = np.linalg.inv(T_cam_from_tag)

    cam_pos = T_world_from_cam[:3, 3]
    cam_rot = T_world_from_cam[:3, :3]  # rotation: camera -> world

    # OpenCV: camera looks along +Z; image-up is -Y.
    forward_world = cam_rot @ np.array([0.0, 0.0, 1.0])
    up_world = -cam_rot @ np.array([0.0, 1.0, 0.0])

    return cam_pos, forward_world, up_world


def _find_companion(scene_json_path: Path, name: str, max_up: int = 3) -> Path | None:
    """Search the scene_json directory and its ancestors for a file named `name`."""
    p = scene_json_path.parent.resolve()
    for _ in range(max_up):
        candidate = p / name
        if candidate.exists():
            return candidate
        if p.parent == p:
            return None
        p = p.parent
    return None


def _find_source_image(scene_json_path: Path, max_up: int = 3) -> Path | None:
    """Locate the real-camera reference image (scene_capture/image/0.png) near the scene JSON."""
    p = scene_json_path.parent.resolve()
    for _ in range(max_up):
        for sub in ("scene_capture/image/0.png", "image/0.png"):
            c = p / sub
            if c.exists():
                return c
        if p.parent == p:
            return None
        p = p.parent
    return None


def _read_png_size(path: Path) -> tuple[int, int] | None:
    """Read (width, height) from a PNG header without external deps. None on failure."""
    try:
        with open(path, "rb") as f:
            f.seek(16)
            w = int.from_bytes(f.read(4), "big")
            h = int.from_bytes(f.read(4), "big")
        if w > 0 and h > 0:
            return w, h
    except (OSError, ValueError):
        pass
    return None


def _load_intrinsics(path: Path) -> tuple[float, float, float, float] | None:
    """Load (fx, fy, cx, cy) from a 3x3 K matrix or a `[fx, fy, cx, cy]` list file."""
    try:
        raw = Path(path).read_text()
    except OSError:
        return None
    cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    try:
        nums = np.fromstring(cleaned, sep=" ")
    except Exception:
        return None
    if nums.size == 9:
        K = nums.reshape(3, 3)
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if nums.size == 4:
        fx, fy, cx, cy = (float(v) for v in nums)
        return fx, fy, cx, cy
    return None


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


def transform_vertices(vertices, T):
    """Transform vertices by 4x4 transformation matrix."""
    ones = np.ones((vertices.shape[0], 1))
    vertices_homogeneous = np.hstack([vertices, ones])
    transformed = (T @ vertices_homogeneous.T).T
    return transformed[:, :3]


def main():
    parser = argparse.ArgumentParser(description='Visualize transformed scene in world/table frame')
    parser.add_argument('--scene_json', type=str, required=True, 
                       help='Path to scene_output_new.json with transformed poses')
    parser.add_argument('--window_size', type=int, nargs=2, default=[1600, 1200],
                       help='Window size (width height)')
    parser.add_argument('--screenshot', type=str, default=None,
                       help='Save screenshot to this path (optional)')
    parser.add_argument('--camera_frame_json', type=str, default=None,
                       help='Path to camera_frame_pose.json. If provided, the viewer '
                            'matches the real-world camera pose for the reference tag.')
    parser.add_argument('--tag_index', type=int, default=0,
                       help='Reference tag index for camera pose lookup (default: 0)')
    parser.add_argument('--match_camera_fov', action='store_true',
                       help='If set with --camera_frame_json and --cam_k, also match the '
                            'camera vertical FOV.')
    parser.add_argument('--cam_k', type=str, default=None,
                       help='Path to cam_K.txt (3x3 intrinsics). Used with --match_camera_fov.')
    parser.add_argument('--image_size', type=int, nargs=2, default=None,
                       metavar=('W', 'H'),
                       help='Real camera image size in pixels. Used with --match_camera_fov.')
    parser.add_argument('--camera_distance_scale', type=float, default=1.5,
                       help='Multiply the camera-to-focal distance. >1 pulls camera '
                            'back (e.g. 2.0 = twice as far). Default 1.5 pulls back '
                            'a bit from the real camera so objects are not cropped.')
    parser.add_argument('--camera_focal', type=float, nargs=3, default=None,
                       metavar=('X', 'Y', 'Z'),
                       help='Override focal point (world frame, meters). E.g. "0 0 0.05" '
                            'to look at the table origin instead of 1 m along the optical axis.')

    args = parser.parse_args()
    
    # Load scene data
    scene_json_path = Path(args.scene_json)
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)

    # Auto-detect real-camera inputs from sibling files near the scene JSON so the
    # default view matches the real camera instead of an isometric fallback.
    if args.camera_frame_json is None:
        found = _find_companion(scene_json_path, "camera_frame_pose.json")
        if found is not None:
            args.camera_frame_json = str(found)
            print(f"[auto-detect] camera_frame_json: {found}")

    if args.camera_frame_json and args.cam_k is None:
        for name in ("cam_K.txt", "cam_params.txt"):
            found = _find_companion(scene_json_path, name)
            if found is not None:
                args.cam_k = str(found)
                print(f"[auto-detect] cam_k: {found}")
                break

    if args.cam_k and args.image_size is None:
        img_path = _find_source_image(scene_json_path)
        if img_path is not None:
            size = _read_png_size(img_path)
            if size is not None:
                args.image_size = list(size)
                print(f"[auto-detect] image_size {size[0]}x{size[1]} from {img_path}")
        if args.image_size is None:
            intr = _load_intrinsics(Path(args.cam_k))
            if intr is not None:
                _, _, cx, cy = intr
                args.image_size = [int(round(2 * cx)), int(round(2 * cy))]
                print(f"[auto-detect] image_size inferred from intrinsics: {args.image_size}")

    if args.camera_frame_json and args.cam_k and args.image_size and not args.match_camera_fov:
        args.match_camera_fov = True

    # Match window aspect ratio to the real camera so the rendered horizontal FOV
    # is consistent with the real photo (PyVista's view_angle controls vfov only).
    if args.match_camera_fov and args.image_size:
        cam_w, cam_h = args.image_size
        cam_ar = float(cam_w) / float(cam_h)
        win_w, win_h = args.window_size
        if abs(win_w / win_h - cam_ar) > 0.01:
            new_h = int(round(win_w / cam_ar))
            args.window_size = [win_w, new_h]
            print(f"[auto-detect] adjusted window_size to {win_w}x{new_h} to match camera aspect")
    
    # Get base directory (Scene_reconstruction directory)
    if scene_json_path.is_absolute():
        base_dir = scene_json_path.parent
        # Go up to Scene_reconstruction if we're in a subdirectory
        while base_dir.name != 'Scene_reconstruction' and base_dir.parent != base_dir:
            base_dir = base_dir.parent
        if base_dir.name != 'Scene_reconstruction':
            base_dir = Path.cwd()
    else:
        base_dir = Path.cwd()
    
    # Fix relative mesh paths
    for obj_data in scene_data['objects']:
        if 'mesh_path' in obj_data and obj_data['mesh_path']:
            mesh_path = Path(obj_data['mesh_path'])
            if not mesh_path.is_absolute():
                # Try relative to base_dir first
                full_path = base_dir / mesh_path
                if not full_path.exists():
                    # Try relative to scene_json location
                    full_path = scene_json_path.parent / mesh_path
                obj_data['mesh_path'] = str(full_path)
    
    print("=" * 60)
    print("VISUALIZE TRANSFORMED SCENE")
    print("=" * 60)
    print(f"Scene JSON: {scene_json_path}")
    print(f"Scene description: {scene_data.get('scene_description', 'N/A')}")
    print(f"Number of objects: {len(scene_data['objects'])}")
    print("=" * 60)
    
    # Create PyVista plotter
    pv.set_plot_theme('document')
    
    # Use off_screen mode if screenshot is requested
    off_screen = args.screenshot is not None
    plotter = pv.Plotter(window_size=args.window_size, off_screen=off_screen)
    plotter.add_title("Transformed Scene in World/Table Frame (Tag 0)", font_size=16)
    
    # Add table plane at Z=0
    table_plane = pv.Plane(center=[0, 0, 0], direction=[0, 0, 1], 
                          i_size=1.5, j_size=1.5, i_resolution=10, j_resolution=10)
    plotter.add_mesh(table_plane, color='lightgray', opacity=0.3, show_edges=True)
    
    # Process and visualize each object
    print("\n=== Objects in World/Table Frame ===")
    for idx, obj_data in enumerate(scene_data['objects']):
        mesh_path = obj_data.get('mesh_path', '')
        prompt = obj_data['prompt']
        
        # Skip objects without mesh
        if not mesh_path or mesh_path == "":
            print(f"\nObject {idx}: {prompt} - SKIPPED (no mesh)")
            continue
        
        position = np.array(obj_data['pose']['position_m'])
        quat_xyzw = np.array(obj_data['pose']['orientation_quat_xyzw'])
        
        print(f"\nObject {idx}: {prompt}")
        print(f"  Mesh: {Path(mesh_path).name}")
        print(f"  Position: {position}")
        print(f"  Orientation (quat): {quat_xyzw}")
        
        # Load transformed mesh
        if not Path(mesh_path).exists():
            print(f"  WARNING: Mesh not found at {mesh_path}")
            continue
        
        vertices, faces, colors = load_obj_file(mesh_path)
        print(f"  Loaded: {len(vertices)} vertices, {len(faces)} faces")
        
        # Create transformation matrix from pose
        R = quat_xyzw_to_matrix(quat_xyzw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        # Transform vertices to world frame
        transformed_vertices = transform_vertices(vertices, T)
        
        # Print Z-range
        z_min = transformed_vertices[:, 2].min()
        z_max = transformed_vertices[:, 2].max()
        print(f"  Z-range in world: [{z_min:.4f}, {z_max:.4f}]")
        
        # Create PyVista mesh
        mesh = pv.PolyData(transformed_vertices)
        
        # Add mesh with vertex colors if available
        if colors is not None:
            # Convert to 0-255 range
            if colors.max() <= 1.0:
                rgb_colors = (colors * 255).astype(np.uint8)
            else:
                rgb_colors = colors.astype(np.uint8)
            mesh['RGB'] = rgb_colors
            plotter.add_mesh(mesh, scalars='RGB', rgb=True, point_size=5,
                           render_points_as_spheres=True, opacity=1.0,
                           label=f'{idx}: {prompt[:30]}')
        else:
            # Fallback to solid color
            fallback_colors = ['lightcoral', 'lightgreen', 'lightblue', 'yellow', 
                             'magenta', 'cyan', 'orange']
            color = fallback_colors[idx % len(fallback_colors)]
            plotter.add_mesh(mesh, color=color, opacity=0.8, show_edges=False,
                           label=f'{idx}: {prompt[:30]}')
        
        # Add small coordinate frame at object position
        obj_arrows = []
        arrow_scale = 0.05
        for direction, arrow_color in [([1,0,0], 'red'), ([0,1,0], 'green'), ([0,0,1], 'blue')]:
            arrow_dir = R @ np.array(direction)
            arrow = pv.Arrow(start=position, direction=arrow_dir, scale=arrow_scale)
            plotter.add_mesh(arrow, color=arrow_color, opacity=0.5)
    
    # # Set camera view
    # plotter.camera_position = 'iso'
    # Set camera view
    if args.camera_frame_json:
        focal_override = tuple(args.camera_focal) if args.camera_focal else None
        cam_pos, forward_world, up_world = load_real_camera_view(
            args.camera_frame_json,
            tag_index=args.tag_index,
        )

        # Decide focal point: explicit override, or 1 m along the optical axis.
        if focal_override is not None:
            focal = np.array(focal_override, dtype=float)
        else:
            focal = cam_pos + forward_world

        # Pull the camera back along the eye→focal axis by distance_scale.
        eye = focal + (cam_pos - focal) * float(args.camera_distance_scale)

        cam_spec = [tuple(eye), tuple(focal), tuple(up_world)]
        plotter.camera_position = cam_spec
        print(f"\nCamera view: matched to real camera (tag {args.tag_index})")
        print(f"  Eye:     ({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f})")
        print(f"  Focal:   ({focal[0]:.3f}, {focal[1]:.3f}, {focal[2]:.3f})")
        print(f"  Up:      ({up_world[0]:.3f}, {up_world[1]:.3f}, {up_world[2]:.3f})")
        print(f"  Distance scale: {args.camera_distance_scale}")
        print(f"\nCamera view: matched to real camera (tag {args.tag_index})")

        # Optional: match the camera's vertical FOV using intrinsics.
        if args.match_camera_fov:
            if not args.cam_k or not args.image_size:
                print("  WARNING: --match_camera_fov requires both --cam_k and --image_size; skipping FOV match.")
            else:
                intr = _load_intrinsics(Path(args.cam_k))
                if intr is None:
                    print(f"  WARNING: Could not parse intrinsics from {args.cam_k}; skipping FOV match.")
                else:
                    _, fy, _, _ = intr
                    img_h = float(args.image_size[1])
                    vfov_deg = float(np.degrees(2.0 * np.arctan(img_h / (2.0 * fy))))
                    plotter.camera.view_angle = vfov_deg
                    print(f"  Vertical FOV: {vfov_deg:.2f} deg (from fy={fy:.1f}, H={img_h:.0f})")
    else:
        plotter.camera_position = 'iso'
    
    # Only add legend if there are labeled items
    if len([obj for obj in scene_data['objects'] if obj.get('mesh_path')]) > 0:
        try:
            plotter.add_legend(size=(0.2, 0.3), loc='upper right')
        except:
            pass  # Skip legend if no labels
    
    # Save screenshot if requested
    if args.screenshot:
        screenshot_path = Path(args.screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(screenshot_path))
        print(f"\nScreenshot saved to: {screenshot_path}")
        plotter.close()
    else:
        # Show interactive plot
        print("\n" + "=" * 60)
        print("Showing interactive 3D visualization...")
        print("=" * 60)
        plotter.show()


if __name__ == "__main__":
    main()
