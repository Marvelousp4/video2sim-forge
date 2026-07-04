from scipy.spatial.transform import Rotation as R
import numpy as np
from camera.april_tags_detection import Customized_Detector
from camera.point_calibration import compute_rigid_transform, draw_axes
import matplotlib.pyplot as plt
import os
import cv2
import datetime

# Camera parameters
teleop_camera_sn = "337122071053"
teleop_camera_intric = [385.383, 385.383, 317.368, 243.951]
data_collection_camera_sn = "242422303080"
data_collection_camera_intric = [390.878, 390.878, 324.335, 245.902]
image_fps = 30
tag_size = 0.07

camera_detector = Customized_Detector(
    serial_number=data_collection_camera_sn,
    camera_intrinsic=data_collection_camera_intric,
    image_fps=image_fps,
    tag_size=tag_size,
    visualize=False
)

index = 0
recording = False
video_writer = None
depth_video_writer = None

while True:
    images = camera_detector.read_images()

    if images is not None:
        color_image = images[0]
        depth_image = images[1]
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image.copy(), alpha=0.5), cv2.COLORMAP_JET)

        # Display image always
        combined = np.hstack((
            cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0])) if color_image.shape != depth_colormap.shape else color_image,
            depth_colormap
        ))
        cv2.imshow('RealSense', combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n') and not recording:
            print("Started recording...")
            recording = True

            # Timestamp and folder setup
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            root_folder = f"Data/run_{timestamp}"
            image_folder = os.path.join(root_folder, 'image')
            depth_folder = os.path.join(root_folder, 'depth')
            cam_intrinsic_path = os.path.join(root_folder, 'cam_params.txt')
            video_path = os.path.join(root_folder, 'color_video.mp4')
            depth_video_path = os.path.join(root_folder, 'depth_video.mp4')

            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(depth_folder, exist_ok=True)

            with open(cam_intrinsic_path, 'w') as f:
                f.write('{}'.format(data_collection_camera_intric))

            height, width = color_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, image_fps, (width, height))
            depth_video_writer = cv2.VideoWriter(depth_video_path, fourcc, image_fps, (depth_colormap.shape[1], depth_colormap.shape[0]))

        if recording:
            # Save images
            cv2.imwrite(os.path.join(image_folder, f"{index}.png"), color_image)
            cv2.imwrite(os.path.join(depth_folder, f"{index}.png"), depth_image)

            # Write video frames
            video_writer.write(color_image)
            depth_video_writer.write(depth_colormap)
            index += 1

        if key == ord('q'):
            print("Exiting...")
            break

# Cleanup
if video_writer:
    video_writer.release()
if depth_video_writer:
    depth_video_writer.release()
cv2.destroyAllWindows()