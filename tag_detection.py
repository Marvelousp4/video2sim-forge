# AprilTag Detection System
from scipy.spatial.transform import Rotation as R
import numpy as np
from camera.april_tags_detection import AprilTagDetector
import time
import signal
import sys

print("=" * 60)
print("AprilTag Detection System - Relative Pose Mode")
print("=" * 60)

# Camera configurations
TELEOP_CAMERA_SN = "337122071053"
DATA_CAMERA_SN = "242422303080"
TELEOP_INTRINSICS = [385.383, 385.383, 317.368, 243.951]
DATA_INTRINSICS = [390.878, 390.878, 324.335, 245.902]
IMAGE_FPS = 30
TAG_SIZE = 0.054
REFERENCE_TAG_ID = 0

print(f"[INFO] Camera configuration:")
print(f"[INFO] - camera SN: {DATA_CAMERA_SN}")
print(f"[INFO] - Tag size: {TAG_SIZE}m, Target FPS: {IMAGE_FPS}")
print(f"[INFO] - Reference tag ID: {REFERENCE_TAG_ID} (coordinate origin)")

print("\n[INFO] Initializing AprilTag detector...")
# Use None for camera_intrinsic to let camera determine actual intrinsics
detector = AprilTagDetector(
    serial_number=DATA_CAMERA_SN,
    camera_intrinsic=None,
    image_fps=IMAGE_FPS,
    tag_size=TAG_SIZE,
    reference_tag_id=REFERENCE_TAG_ID
)
print("[INFO] AprilTag detector initialized successfully")

print("\n[INFO] Starting detection loop, press Ctrl+C to exit...")
print("=" * 60)

detection_count = 0
last_detection_time = time.time()
first_frame_saved = False

while True:
    start_time = time.time()
    results = detector.detect()
        
    detection_time = time.time() - start_time
    
    current_time = time.time()
    time_since_last = current_time - last_detection_time
    last_detection_time = current_time
    
    if results is not None:
        detection_count += 1
        print(f"\n[Detection #{detection_count}] Interval: {time_since_last:.3f}s, Processing: {detection_time:.3f}s")
            
    else:
        if detection_count == 0:
            print(f"[INFO] Waiting for AprilTags... (running {time_since_last:.1f}s)")
        elif time_since_last > 2.0:
            print(f"[INFO] No tags detected (last detection: {time_since_last:.1f}s ago)")
            
    time.sleep(0.05)
