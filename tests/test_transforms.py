import importlib.util
import json
from pathlib import Path

import numpy as np


def load_transform_module():
    module_path = Path("scripts/step4_export_transforms.py")
    spec = importlib.util.spec_from_file_location("step4_export_transforms", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transform_to_tag_frame_applies_inverse_tag_pose(tmp_path: Path):
    transforms = load_transform_module()
    camera_frame_json = tmp_path / "camera_frame_pose.json"
    camera_frame_json.write_text(
        json.dumps(
            [
                {
                    "tag_index": 0,
                    "position(m)": [1.0, 2.0, 3.0],
                    "orientation_deg_XYZ(deg)": [0.0, 0.0, 0.0],
                }
            ]
        )
    )

    position, quat = transforms.transform_to_tag_frame(
        np.array([2.0, 4.0, 6.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        camera_frame_json,
        set_z_zero=False,
    )

    assert np.allclose(position, [1.0, 2.0, 3.0])
    assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0])


def test_transform_to_tag_frame_can_project_to_table_plane(tmp_path: Path):
    transforms = load_transform_module()
    camera_frame_json = tmp_path / "camera_frame_pose.json"
    camera_frame_json.write_text(
        json.dumps(
            [
                {
                    "tag_index": 0,
                    "position(m)": [1.0, 2.0, 3.0],
                    "orientation_deg_XYZ(deg)": [0.0, 0.0, 0.0],
                }
            ]
        )
    )

    position, _ = transforms.transform_to_tag_frame(
        np.array([2.0, 4.0, 6.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        camera_frame_json,
        set_z_zero=True,
    )

    assert np.allclose(position, [1.0, 2.0, 0.0])

