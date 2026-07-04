import json
from pathlib import Path


DEMO_OUTPUTS = Path("examples/ubuntu_demo/outputs")


def load_json(name: str):
    return json.loads((DEMO_OUTPUTS / name).read_text())


def test_ubuntu_demo_timing_records_gpu_pipeline_completion():
    timing = load_json("pipeline_timing.json")
    steps = {step["name"]: step for step in timing["steps"]}

    assert timing["total_s"] > 60
    assert steps["Step 1: Gemini"]["status"] == "skipped"
    assert steps["Step 2: SAM3"]["status"] == "completed"
    assert steps["Step 3: SAM3D"]["status"] == "completed"
    assert steps["Step 4: Assemble Output"]["status"] == "completed"
    assert steps["Step 5: Export Transforms"]["status"] == "completed"
    assert steps["Step 6: Visualize"]["status"] == "completed"
    assert steps["Step 7: OBJ to URDF"]["status"] == "completed"


def test_ubuntu_demo_final_scene_has_simulation_assets():
    scene = load_json("scene_output_final.json")

    assert scene["manipulated_object"] == "orange fruit"
    assert scene["target_object"] == "light blue bowl"
    assert len(scene["objects"]) == 2
    for obj in scene["objects"]:
        assert obj["reconstruction_status"] == "ok"
        assert obj["pose"]["position_m"][2] == 0.0
        assert obj["urdf_path"]
        assert obj["mass"] > 0
        assert obj["volume_m3"] > 0
        assert obj["inertia"]["ixx"] > 0
        assert obj["inertia"]["iyy"] > 0
        assert obj["inertia"]["izz"] > 0


def test_ubuntu_demo_media_and_logs_are_present():
    expected_files = [
        "environment.md",
        "validate_config.log",
        "run.log",
        "mask_000.png",
        "mask_001.png",
        "final_scene_visualization.png",
        "urdfs/obj_0000.urdf",
        "urdfs/obj_0001.urdf",
    ]

    for name in expected_files:
        path = DEMO_OUTPUTS / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_ubuntu_demo_fixture_is_sanitized():
    for path in DEMO_OUTPUTS.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".log", ".md", ".urdf"}:
            text = path.read_text()
            assert "/home/linhaobai" not in text
            assert "/tmp/video2sim-forge" not in text
            assert "sk-" not in text

