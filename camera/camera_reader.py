import pyrealsense2 as rs
import numpy as np
import cv2
import sys


'''
To use the visual teleoperation for data collection -
we have two cameras: 
    1. D455: used for collecting robot demo and robot policy rollout
    serial number: 242422303080
    if image_res = 640 * 480, fx = fy = 390.878, cx = 324.335, cy = 245.902
    2. D435: used for visual teleoperation
    serial number: 337122071053
    if image_res = 640 * 480, fx = fy = 385.383, cx = 317.368, cy = 243.951
'''

class BGR_Reader:
    def __init__(self, width=640, height=480, fps=30, visualize=False, serial_number = None, depth=True):
        """
        Initialize the RGB-D reader. For teleoperation.
        
        Args:
            width (int): The desired width of the color and depth streams.
            height (int): The desired height of the color and depth streams.
            fps (int): The desired frame rate of the streams.
            visualize (bool): If True, display frames in a window.
            depth (bool): If True, enable depth streaming.
        """
        print(f"[INFO] Initializing BGR reader...")
        print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}")
        print(f"[INFO] Visualization: {'ON' if visualize else 'OFF'}, Depth: {'ON' if depth else 'OFF'}")
        print(f"[INFO] Camera SN: {serial_number or 'Auto-detect'}")
        self.width = width
        self.height = height
        self.fps = fps
        self.visualize = visualize
        self.depth_enabled = depth
        self.serial_number = serial_number

        self.pipeline = None    
        self.align = None
        self._is_running = False

    def start(self):
        """
        Configure and start the RealSense pipeline for color and optional depth streaming.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()

        ctx = rs.context()
        devices = list(ctx.devices)
        if len(devices) == 0:
                raise RuntimeError("No RealSense devices found")
            
        print(f"[INFO] Found {len(devices)} RealSense device(s):")
        target_device = None
        for i, dev in enumerate(devices):
            device_name = dev.get_info(rs.camera_info.name)
            device_serial = dev.get_info(rs.camera_info.serial_number)
            print(f"[INFO] Device {i+1}: {device_name}, SN: {device_serial}")
            if self.serial_number == device_serial:
                target_device = dev
                print(f"[INFO] Target camera found: {device_name} (SN: {device_serial})")
                break
        if target_device is None:
            if self.serial_number:
                raise RuntimeError(f"Camera with SN {self.serial_number} not found")
            else:
                target_device = devices[0]
                self.serial_number = target_device.get_info(rs.camera_info.serial_number)
                print(f"[INFO] No SN specified, using first device: SN {self.serial_number}")
                
        config.enable_device(self.serial_number)        # ← key line

        # Enable the color stream
        print(f"[INFO] Enabling color stream: {self.width}x{self.height} @ {self.fps}fps")

        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Enable the depth stream if selected
        if self.depth_enabled:
            print(f"[INFO] Enabling depth stream: {self.width}x{self.height} @ {self.fps}fps")

            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            print("[INFO] Starting pipeline...")
            profile = self.pipeline.start(config)
            self._is_running = True
            print("[INFO] RealSense pipeline started successfully")
            
            # Get actual intrinsics from active profile
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            print(f"[INFO] Active intrinsics: fx={self.color_intrinsics.fx:.1f}, fy={self.color_intrinsics.fy:.1f}, "
                  f"cx={self.color_intrinsics.ppx:.1f}, cy={self.color_intrinsics.ppy:.1f}")
            
            # Get depth scale for unit conversion
            if self.depth_enabled:
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f"[INFO] Depth scale: {self.depth_scale:.6f} (meters per depth unit)")

        if self.depth_enabled:
            # Align depth to color frame
            self.align = rs.align(rs.stream.color)
            print("[INFO] Depth alignment configured")

        if self.visualize:
                print("[INFO] Visualization enabled, press 'q' to exit")

    def get_intrinsics(self):
        if hasattr(self, 'color_intrinsics'):
            return [self.color_intrinsics.fx, self.color_intrinsics.fy, 
                   self.color_intrinsics.ppx, self.color_intrinsics.ppy]
        else:
            print("[WARNING] Intrinsics not available, using default values")
            return [385.4, 385.4, 317.4, 244.0]  # Fallback values
    def get_depth_scale(self):
        """Get depth scale for unit conversion"""
        return getattr(self, 'depth_scale', 0.001)  # Default 1mm per unit

    def read(self):
        """
        Returns the latest RGB and Depth frames as NumPy arrays. 
        
        Returns:
            tuple: (color_image, depth_image)
                   - color_image: RGB frame as a NumPy array.
                   - depth_image: Depth frame as a NumPy array (if depth mode is enabled).
        """
        if not self._is_running or self.pipeline is None:
            print("[ERROR] Pipeline not running or not initialized")

            return None, None

        # Wait for a new frame
        frames = self.pipeline.wait_for_frames()

        if self.depth_enabled:
            # Align depth to color frame
            frames = self.align.process(frames)

        # Get color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[WARNING] No color frame received")
            return None, None

        color_image = np.asanyarray(color_frame.get_data())

        # Get depth frame (if enabled)
        depth_image = None
        if self.depth_enabled:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                print("[WARNING] No depth frame received")

        # Show frame if visualization is on
        if self.visualize:
            cv2.imshow("RGB-D Frame", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end()
                return None, None

        return color_image, depth_image

    def end(self):
        """
        Stop the pipeline and release resources.
        """
        if self.pipeline and self._is_running:
            self.pipeline.stop()
        self._is_running = False
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped and resources released")

class VideoReader:
    def __init__(self, width=640, height=480, fps=30, visualize=False):
        """
        Initialize the Video reader. For robot manipulation.
        
        Args:
            width (int): The desired width of the color stream.
            height (int): The desired height of the color stream.
            fps (int): The desired frame rate of the color stream.
            visualize (bool): If True, display frames in a window.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.visualize = visualize
        
        self.pipeline = None
        self._is_running = False

    def start(self):
        """
        Configure and start the RealSense pipeline for color streaming only.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        self._is_running = True
        if self.visualize:
            print("[INFO] Visualization is ON. Press 'q' to exit.")

    def read(self):
        """
        Returns the latest RGB frame as a NumPy array. If visualize=True, also shows it on screen.
        
        Returns:
            np.ndarray: The RGB frame, or None if no frame was retrieved or if 'q' was pressed.
        """
        if not self._is_running or self.pipeline is None:
            return None

        # Wait for a new frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        # Convert frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Show frame if visualization is on
        if self.visualize:
            cv2.imshow("RGB Frame", color_image)
            # If 'q' is pressed, end immediately
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end()
                return None

        return color_image

    def end(self):
        """
        Stop the pipeline and release resources.
        """
        if self.pipeline and self._is_running:
            self.pipeline.stop()
        self._is_running = False
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped and resources released.")

# Example usage
if __name__ == "__main__":
    reader = BGR_Reader(visualize=True, depth=True)
    reader.start()

    while True:
        color_image, depth_image = reader.read()
        if color_image is None:
            break