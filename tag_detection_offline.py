# from robot.allegro import Allegro
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys, os, glob, json, cv2
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation
ROBOT_TRANSFORM = np.array([
    [0, -1,  0],   # robot_x = -april_y
    [-1, 0,  0],   # robot_y = -april_x
    [0,  0, -1]    # robot_z = -april_z
], dtype=np.float32)

def april_to_robot_axes_in_camera(t_cam_april, R_cam_april):
    R_cam_robot = R_cam_april @ ROBOT_TRANSFORM.T
    return t_cam_april, R_cam_robot

def draw_robot_axes(overlay, camera_params, tag_size, R_cam_april, t_cam, center, tag_id):
    fx, fy, cx, cy = camera_params
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=float)
    robot_axes_in_tag = (ROBOT_TRANSFORM.T @ np.eye(3)) * float(tag_size)  # 3x3
    axes_in_cam = R_cam_april @ robot_axes_in_tag                          # 3x3

    ipoints, _ = cv2.projectPoints(
        axes_in_cam.T.reshape(-1,1,3),
        np.zeros(3, dtype=float),
        t_cam.reshape(3,1).astype(float),
        K, np.zeros(5)
    )
    ipoints = np.round(ipoints).astype(int)
    c = tuple(np.round(center).astype(int).ravel())
    colors = [(0,0,255),(0,255,0),(255,0,0)]  # X,Y,Z
    cv2.line(overlay, c, tuple(ipoints[0].ravel()), colors[0], 3)
    cv2.line(overlay, c, tuple(ipoints[1].ravel()), colors[1], 3)
    cv2.line(overlay, c, tuple(ipoints[2].ravel()), colors[2], 3)
    cv2.putText(overlay, f"{tag_id}", (c[0]+10, c[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
teleop_camera_sn = "337122071053"
teleop_camera_intric_fallback = [385.383, 385.383, 317.368, 243.951]
data_collection_camera_sn = "242422303080"
data_collection_camera_intric_fallback = [390.878, 390.878, 324.335, 245.902]
image_fps = 30
tag_size = 0.054
REFERENCE_TAG_ID = 0

def get_live_intrinsics_or_fallback():
    try:
        from camera.april_tags_detection import AprilTagDetector as LiveAT
        temp = LiveAT(
            serial_number=data_collection_camera_sn,
            camera_intrinsic=None,
            image_fps=image_fps, visualize=False, depth=False,
            tag_size=tag_size, reference_tag_id=REFERENCE_TAG_ID
        )
        intr = list(map(float, temp.camera_intrinsic))
        try:
            temp.reader.end()
        except Exception:
            pass
        print(f"[INFO] Using LIVE intrinsics: fx={intr[0]:.1f}, fy={intr[1]:.1f}, cx={intr[2]:.1f}, cy={intr[3]:.1f}")
        return intr
    except Exception as e:
        intr = data_collection_camera_intric_fallback
        print(f"[WARN] Live intrinsics failed ({e}), fallback to: {intr}")
        return intr

if len(sys.argv) < 2:
    print("Usage: python script.py <images_folder>")
    sys.exit(1)
folder_path = sys.argv[1]

image_files_numeric = sorted(
    [f for f in os.listdir(folder_path)
     if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')) and os.path.splitext(f)[0].isdigit()],
    key=lambda f: int(os.path.splitext(f)[0])
)
if len(image_files_numeric) > 0:
    image_files = image_files_numeric
else:
    image_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(folder_path, "color_*.jpg"))])

if not image_files:
    print(f"No images found in {folder_path}. Expected numeric filenames or 'color_*.jpg'.")
    sys.exit(1)

images = []
for fname in image_files:
    img = cv2.imread(os.path.join(folder_path, fname))
    if img is not None:
        images.append(img)
    else:
        print(f"[WARN] Failed to load {fname}")

print(f"[INFO] Loaded {len(images)} images from {folder_path}")
camera_params = get_live_intrinsics_or_fallback()

at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

for i, color in enumerate(images):
    vis = color.copy()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    tags = at_detector.detect(gray,
                              estimate_tag_pose=True,
                              camera_params=camera_params,
                              tag_size=tag_size)

    if len(tags) == 0:
        print(f"[INFO] No tags in {image_files[i]}")
        with open(os.path.join(folder_path, f"camera_frame_img_{i}.json"), "w") as f:
            json.dump([], f, indent=2)
        with open(os.path.join(folder_path, f"reference_frame_img_{i}.json"), "w") as f:
            json.dump([], f, indent=2)
        continue

    print(f"[INFO] {len(tags)} tags in {image_files[i]}")

    per_frame = []

    for j, tag in enumerate(tags):
        t_cam = tag.pose_t.squeeze(1)   #  (3,)
        R_cam_april = tag.pose_R        #  (3,3) (Tag-April -> Cam)

        p_cam, R_cam_robot = april_to_robot_axes_in_camera(t_cam, R_cam_april)

        draw_robot_axes(vis, camera_params, 0.05, R_cam_april, tag.pose_t, tag.center, tag.tag_id)

        per_frame.append({
            'j': j,
            'id': int(tag.tag_id),
            'p_cam': p_cam,
            'R_cam_robot': R_cam_robot,
            'R_cam_april': R_cam_april,
            'center': tag.center
        })

    cam_array = []
    for d in per_frame:
        eul = Rotation.from_matrix(d['R_cam_robot']).as_euler('XYZ', degrees=True)
        cam_array.append({
            "tag_index": int(d['id']),
            "position(m)": [round(float(x), 5) for x in d['p_cam']],
            "orientation_deg_XYZ(deg)": [round(float(a), 3) for a in eul]
        })

    cam_json_path = os.path.join(folder_path, f"camera_frame_img_{i}.json")
    with open(cam_json_path, "w") as f:
        json.dump(cam_array, f, indent=2)
    print(f"[SAVE] {cam_json_path}")

    ref_array = []
    ref = next((d for d in per_frame if d['id'] == REFERENCE_TAG_ID), None)
    if ref is None:
        print(f"[WARN] Reference tag {REFERENCE_TAG_ID} not found in frame {i}; saving empty reference array.")
    else:
        # T_C_ref
        T_C_ref = np.eye(4)
        T_C_ref[:3,:3] = ref['R_cam_robot']
        T_C_ref[:3, 3] = ref['p_cam']
        T_ref_C = np.linalg.inv(T_C_ref)

        for d in per_frame:
            # T_C_tag
            T_C_tag = np.eye(4)
            T_C_tag[:3,:3] = d['R_cam_robot']
            T_C_tag[:3, 3] = d['p_cam']

            # T_ref_tag
            T_ref_tag = T_ref_C @ T_C_tag
            rel_p = T_ref_tag[:3,3]
            rel_R = T_ref_tag[:3,:3]

            eul_rel = Rotation.from_matrix(rel_R).as_euler('XYZ', degrees=True)
            ref_array.append({
                "tag_index": int(d['id']),
                "position(m)": [round(float(x), 5) for x in rel_p],
                "orientation_deg_XYZ(deg)": [round(float(a), 3) for a in eul_rel]
            })

    ref_json_path = os.path.join(folder_path, f"reference_frame_img_{i}.json")
    with open(ref_json_path, "w") as f:
        json.dump(ref_array, f, indent=2)
    print(f"[SAVE] {ref_json_path}")

    cv2.namedWindow('Apriltag_detector', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Apriltag_detector', vis)
    save_path = os.path.join(folder_path, f"{i}_tag_view.png")
    cv2.imwrite(save_path, vis)
    print(f"[SAVE] {save_path}")

    while True:
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key & 0xFF == ord('n'):
            cv2.destroyWindow('Apriltag_detector')
            break
        if cv2.getWindowProperty('Apriltag_detector', cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()