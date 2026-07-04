import json
from pathlib import Path

import pytest

from run_pipeline import _format_duration, assemble_final_output, require_file


def test_format_duration_seconds_minutes_hours():
    assert _format_duration(12.345) == "12.35s"
    assert _format_duration(75.0) == "1m 15.0s"
    assert _format_duration(3675.0) == "1h 1m 15.0s"


def test_assemble_final_output_supports_rich_mask_mapping(tmp_path: Path):
    (tmp_path / "gemini_scene.json").write_text(
        json.dumps(
            {
                "scene_description": "A cube is pushed toward a bowl.",
                "task_type": "push",
                "manipulated_prompt": "red cube",
                "target_prompt": "blue bowl",
                "objects": [
                    {"prompt": "red cube", "material_type": "plastic"},
                    {"prompt": "blue bowl", "material_type": "ceramic"},
                ],
            }
        )
    )
    (tmp_path / "sam3d_results.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": 0,
                        "mesh_path": str(tmp_path / "obj_0000.obj"),
                        "pose": {
                            "position_m": [0.1, 0.2, 0.3],
                            "orientation_quat_xyzw": [0, 0, 0, 1],
                        },
                        "reconstruction_status": "success",
                    },
                    {
                        "object_id": 1,
                        "mesh_path": "obj_0001.obj",
                        "pose": None,
                        "reconstruction_status": "failed",
                    },
                ]
            }
        )
    )
    (tmp_path / "mask_to_prompt_mapping.json").write_text(
        json.dumps(
            {
                "0": {"prompt_idx": 0, "prompt": "red cube"},
                "1": {"prompt_idx": 1, "prompt": "blue bowl"},
            }
        )
    )

    output = assemble_final_output(tmp_path)

    assert output["task_type"] == "object push"
    assert output["target_object"] == "blue bowl"
    assert output["target_id"] == 1
    assert output["objects"][0]["is_manipulated"] is True
    assert output["objects"][0]["mesh_path"] == "obj_0000.obj"
    assert output["objects"][1]["pose"]["position_m"] == [0, 0, 0]


def test_require_file_accepts_existing_file(tmp_path: Path):
    capture_file = tmp_path / "scene_capture" / "image" / "0.png"
    capture_file.parent.mkdir(parents=True)
    capture_file.write_bytes(b"not a real image")

    require_file(capture_file, "scene RGB frame")


def test_require_file_exits_for_missing_file(tmp_path: Path):
    with pytest.raises(SystemExit) as exc_info:
        require_file(tmp_path / "missing.png", "scene RGB frame")

    assert exc_info.value.code == 1
