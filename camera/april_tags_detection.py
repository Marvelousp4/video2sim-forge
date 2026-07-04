import pyrealsense2 as rs
import numpy as np
import cv2
import sys
ROBOT_TRANSFORM = np.array([
    [0, -1, 0],   # robot_x = -april_y
    [-1, 0, 0],   # robot_y = -april_x  
    [0, 0, -1]    # robot_z = -april_z
], dtype=np.float32)

from dt_apriltags import Detector
import os
from scipy.spatial.transform import Rotation

def draw_robot_coordinate_axes(overlay, camera_params, tag_size, original_pose_matrix, center, index):
    fx, fy, cx, cy = camera_params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)

    # camera←tag
    R_tag_to_cam = original_pose_matrix[:3, :3]
    tvec = original_pose_matrix[:3, 3]

    # v_tag = T^T * e_robot
    T = ROBOT_TRANSFORM
    robot_axes_in_tag = (T.T @ np.eye(3)) * float(tag_size)   # (3x3)

    # X_cam = R * X_tag
    axes_in_camera = R_tag_to_cam @ robot_axes_in_tag          # (3x3)

    axes_in_camera_3d = axes_in_camera.T.reshape(-1, 1, 3)

    ipoints, _ = cv2.projectPoints(axes_in_camera_3d,
                                   np.zeros(3, dtype=float),
                                   tvec.reshape(3, 1).astype(float),
                                   K, np.zeros(5))
    ipoints = np.round(ipoints).astype(int)

    center = tuple(np.round(center).astype(int).ravel())

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    cv2.line(overlay, center, tuple(ipoints[0].ravel()), colors[0], 3)  # Rx
    cv2.line(overlay, center, tuple(ipoints[1].ravel()), colors[1], 3)  # Ry
    cv2.line(overlay, center, tuple(ipoints[2].ravel()), colors[2], 3)  # Rz

    if index is not None:
        cv2.putText(overlay, f"{index}", (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
from camera.camera_reader import BGR_Reader
def transform_to_robot_frame(position, rotation_matrix):
    """Transform pose from AprilTag frame to robot frame"""
    # Transform position using coordinate transformation
    robot_position = ROBOT_TRANSFORM @ position
    
    # Transform rotation: R_robot = T * R_april * T^-1
    robot_rotation = ROBOT_TRANSFORM @ rotation_matrix @ ROBOT_TRANSFORM.T
    
    return robot_position, robot_rotation
class AprilTagDetector():
    def __init__(self, serial_number, camera_intrinsic, image_fps, visualize=True, depth=True, tag_size=0.055, reference_tag_id=0):
        print(f"[INFO] Initializing AprilTag detector...")
        print(f"[INFO] Camera SN: {serial_number}")
        if camera_intrinsic is not None:
            print(f"[INFO] Provided intrinsics: fx={camera_intrinsic[0]:.1f}, fy={camera_intrinsic[1]:.1f}, cx={camera_intrinsic[2]:.1f}, cy={camera_intrinsic[3]:.1f}")
        else:
            print(f"[INFO] Will use camera's actual intrinsics")
        print(f"[INFO] Tag size: {tag_size}m, FPS: {image_fps}, Reference tag ID: {reference_tag_id}")
        self.visualize = visualize
        self.depth = depth
        self.serial_number = serial_number
        self.camera_intrinsic = camera_intrinsic
        self.tag_size = tag_size
        self.image_fps = image_fps
        self.reference_tag_id = reference_tag_id
        self.reader = BGR_Reader(visualize=self.visualize, depth=self.depth, serial_number=self.serial_number, fps=self.image_fps)
        self.reader.start()
        if self.camera_intrinsic is None:
            self.camera_intrinsic = self.reader.get_intrinsics()
            print(f"[INFO] Using actual camera intrinsics: fx={self.camera_intrinsic[0]:.1f}, fy={self.camera_intrinsic[1]:.1f}, "
                    f"cx={self.camera_intrinsic[2]:.1f}, cy={self.camera_intrinsic[3]:.1f}")
        else:
            print(f"[INFO] Using provided intrinsics: fx={self.camera_intrinsic[0]:.1f}, fy={self.camera_intrinsic[1]:.1f}, "
                    f"cx={self.camera_intrinsic[2]:.1f}, cy={self.camera_intrinsic[3]:.1f}")
        
        self.depth_scale = self.reader.get_depth_scale()
        print(f"[INFO] Depth scale: {self.depth_scale:.6f} meters per unit")
        print("[INFO] Camera reader initialized successfully")
        self.detector = Detector(families='tagStandard41h12', nthreads=1,
                                quad_decimate=1.0, quad_sigma=0.0, refine_edges=1,
                                decode_sharpening=0.25, debug=0)
    def calculate_reprojection_rmse(self, tag, rotation_matrix, position):
        """Calculate reprojection RMSE with robust 8-way corner matching"""
        half_size = self.tag_size / 2
        tag_corners_3d = np.array([
            [-half_size, -half_size, 0],  # Bottom-left
            [ half_size, -half_size, 0],  # Bottom-right  
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0]   # Top-left
        ])
        
        # Use OpenCV projectPoints for consistency
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        tvec = position.reshape(3, 1)
        fx, fy, cx, cy = self.camera_intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        dcoeffs = np.zeros(5)  # No distortion for now
        
        proj, _ = cv2.projectPoints(tag_corners_3d, rvec, tvec, K, dcoeffs)
        projected_corners = proj.reshape(-1, 2)
        
        # Get detected corners
        detected_corners = np.array(tag.corners).reshape(4, 2)
        
        # Try all 8 possible corner matchings (4 rotations × 2 directions)
        min_rmse = float('inf')
        
        for direction in [1, -1]:  # Forward and reverse
            for offset in range(4):  # 4 rotation offsets
                if direction == 1:
                    # Forward direction
                    matched_projected = np.roll(projected_corners, offset, axis=0)
                else:
                    # Reverse direction (flip + roll)
                    matched_projected = np.roll(projected_corners[::-1], offset, axis=0)
                
                # Calculate RMSE for this matching
                errors = matched_projected - detected_corners
                rmse = np.sqrt(np.mean(errors**2))
                min_rmse = min(min_rmse, rmse)
        
        # Optional: Pixel size sanity check
        if position[2] > 0.1:  # Valid depth
            side_measured = 0.5 * (np.linalg.norm(detected_corners[1] - detected_corners[0]) +
                                    np.linalg.norm(detected_corners[2] - detected_corners[1]))
            side_predicted = fx * self.tag_size / position[2]
            
            # Print sanity check occasionally
            import random
            if random.random() < 0.1:  # 10% chance to print
                print(f"[SANITY] Tag {tag.tag_id} side_px: measured={side_measured:.1f}, predicted={side_predicted:.1f}")
        
        return min_rmse
    def enhance_contrast(self, image, alpha=1.1, beta=10):
        """Enhance image contrast for better AprilTag detection"""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
            
    def calculate_relative_poses(self, detected_tags):
        """Calculate poses relative to reference tag coordinate frame using matrices"""
        if not detected_tags:
            return None
            
        # Find reference tag
        reference_tag = None
        for tag_data in detected_tags:
            if tag_data['id'] == self.reference_tag_id:
                reference_tag = tag_data
                break
        
        if reference_tag is None:
            print(f"[WARNING] Reference tag {self.reference_tag_id} not found")
            return detected_tags  # Return absolute poses
        
        # Get reference transformation matrix (absolute pose)
        ref_pose = reference_tag['pose_6d']
        ref_position = ref_pose[:3]
        ref_rotation_matrix = reference_tag['rotation_matrix']
        
        # Create reference transformation matrix T_cam_from_ref
        T_robot_from_ref = np.eye(4)
        T_robot_from_ref[:3, :3] = ref_rotation_matrix
        T_robot_from_ref[:3, 3]  = ref_position
        
        # Inverse to get T_ref_from_cam
        T_ref_from_robot = np.linalg.inv(T_robot_from_ref)

        # Calculate relative poses
        relative_results = []
        for tag_data in detected_tags:
            if tag_data['id'] == self.reference_tag_id:
                # Reference tag is always at origin with identity rotation
                relative_data = tag_data.copy()
                relative_data['pose_6d'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                relative_data['rotation_matrix'] = np.eye(3)  # Identity matrix
                relative_data['quaternion_xyzw'] = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
                relative_results.append(relative_data)
            else:
                # Calculate relative pose using matrix operations
                tag_pose = tag_data['pose_6d']
                tag_position = tag_pose[:3]
                tag_rotation_matrix = tag_data['rotation_matrix']
                
                # Create tag transformation matrix T_cam_from_tag
                T_cam_from_tag = np.eye(4)
                T_cam_from_tag[:3, :3] = tag_rotation_matrix
                T_cam_from_tag[:3, 3] = tag_position

                # Calculate relative transformation: T_ref_from_tag = T_ref_from_robot @ T_cam_from_tag
                T_ref_from_tag = T_ref_from_robot @ T_cam_from_tag

                # Extract relative position and rotation
                relative_position = T_ref_from_tag[:3, 3]
                relative_rotation_matrix = T_ref_from_tag[:3, :3]
                relative_euler = Rotation.from_matrix(relative_rotation_matrix).as_euler('XYZ', degrees=True)
                
                # Create relative result
                relative_data = tag_data.copy()
                relative_data['pose_6d'] = np.concatenate([relative_position, relative_euler])
                relative_data['rotation_matrix'] = relative_rotation_matrix
                relative_data['quaternion_xyzw'] = Rotation.from_matrix(relative_rotation_matrix).as_quat()
                relative_results.append(relative_data)
        
        return relative_results
    
    def read_images(self):
        color_image, depth_image = self.reader.read()
        if color_image is None:
            print("[ERROR] Failed to read image!")
            return None
        return [color_image, depth_image]

    def detect(self, debug=False, save_first_frame=False):
        color_image, depth_image = self.reader.read()
        if color_image is None:
            print("[ERROR] Failed to read image!")
            return None

        enhanced_image = self.enhance_contrast(color_image)

        gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

        tags = self.detector.detect(gray_image, 
                                estimate_tag_pose=True, 
                                camera_params=self.camera_intrinsic, 
                                tag_size=self.tag_size)
        if len(tags) == 0:
                if self.visualize:
                    cv2.namedWindow('AprilTag Detector', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('AprilTag Detector', enhanced_image)
                    cv2.waitKey(1)
                return None
        detected_tags = []
        for tag in tags:
            # Camera
            position = tag.pose_t.squeeze()
            rot_matrix = tag.pose_R
            robot_position, robot_rot_matrix = transform_to_robot_frame(position, rot_matrix)
            euler_deg = Rotation.from_matrix(robot_rot_matrix).as_euler('XYZ', degrees=True)
            depth_raw = None
            if depth_image is not None:
                center_y, center_x = int(tag.center[1]), int(tag.center[0])
                half_size = 3
                y1, y2 = max(0, center_y - half_size), min(depth_image.shape[0], center_y + half_size + 1)
                x1, x2 = max(0, center_x - half_size), min(depth_image.shape[1], center_x + half_size + 1)
                depth_region = depth_image[y1:y2, x1:x2]
                valid_depths = depth_region[depth_region > 0]  # Remove invalid (0) depths
                
                if len(valid_depths) > 0:
                    depth_raw = int(np.median(valid_depths))  # Robust median
            rmse = self.calculate_reprojection_rmse(tag, rot_matrix, position)
            rmse_status = "✓" if rmse <= 1.0 else "⚠"
            if self.visualize:
                draw_robot_coordinate_axes(enhanced_image, self.camera_intrinsic, self.tag_size,
                                np.concatenate([rot_matrix, tag.pose_t], axis=1),
                                tag.center, tag.tag_id)
            robot_quaternion_xyzw = Rotation.from_matrix(robot_rot_matrix).as_quat()  # [x,y,z,w] format
            detected_tags.append({
                'id': tag.tag_id,
                'pose_6d': np.concatenate([robot_position, euler_deg]),  # Robot frame
                'rotation_matrix': robot_rot_matrix,  # Robot frame
                'quaternion_xyzw': robot_quaternion_xyzw,  # Robot frame
                'center': tag.center,
                'depth_raw': depth_raw,
                'rmse_px': rmse,
                'original_position': position,  # Keep original for reference
                'original_rotation_matrix': rot_matrix  # Keep original for reference
            })
        if self.visualize:
            cv2.namedWindow('AprilTag Detector', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('AprilTag Detector', enhanced_image)
            cv2.waitKey(1)
        relative_results = self.calculate_relative_poses(detected_tags)
        # Print results with validation
        print(f"[INFO] Detected {len(tags)} tags")
        for result in relative_results:
            tag_id = result['id']
            pose = result['pose_6d']
            x, y, z, rx, ry, rz = pose
            
            if tag_id == self.reference_tag_id:
                print(f"[INFO] Tag {tag_id} (Reference): Position [0.000, 0.000, 0.000]m, Rotation [0.00°, 0.00°, 0.00°]")
            else:
                print(f"[INFO] Tag {tag_id} (Relative to {self.reference_tag_id}): Position [{x:.3f}, {y:.3f}, {z:.3f}]m, Rotation [{rx:.1f}°, {ry:.1f}°, {rz:.1f}°]")
            
            # Print depth with PnP/Depth ratio validation
            if result['depth_raw'] is not None and result['depth_raw'] > 0:
                depth_meters = result['depth_raw'] * self.depth_scale
                depth_mm = depth_meters * 1000
                
                # For absolute poses, check PnP vs depth consistency
                if tag_id in [tag['id'] for tag in detected_tags]:
                    original_tag = next(tag for tag in detected_tags if tag['id'] == tag_id)
                    pnp_z = original_tag['original_position'][2]  # Use original Z from PnP
                    if pnp_z > 0.1:  # Valid PnP distance
                        ratio = depth_meters / pnp_z
                        status = "✓" if 0.9 <= ratio <= 1.1 else "⚠"
                        print(f"[INFO] Tag {tag_id} depth: {depth_mm:.0f}mm ({depth_meters:.3f}m), center: ({int(result['center'][0])}, {int(result['center'][1])})")
                        print(f"[VALIDATION] {status} PnP/Depth ratio: {ratio:.3f} (PnP: {pnp_z:.3f}m vs Depth: {depth_meters:.3f}m)")
                    else:
                        print(f"[INFO] Tag {tag_id} depth: {depth_mm:.0f}mm ({depth_meters:.3f}m), center: ({int(result['center'][0])}, {int(result['center'][1])})")
                else:
                    print(f"[INFO] Tag {tag_id} depth: {depth_mm:.0f}mm ({depth_meters:.3f}m), center: ({int(result['center'][0])}, {int(result['center'][1])})")
        