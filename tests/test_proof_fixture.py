import json
from pathlib import Path


PROOF_OUTPUTS = Path("examples/proof_run/outputs")


def load_json(name: str):
    return json.loads((PROOF_OUTPUTS / name).read_text())


def test_proof_fixture_scene_outputs_are_coherent():
    scene = load_json("scene_output.json")
    world_scene = load_json("scene_output_new.json")
    final_scene = load_json("scene_output_final.json")

    assert scene["task_type"] == "interactive object manip"
    assert scene["manipulated_object"] == "orange fruit"
    assert scene["target_object"] == "light blue bowl"
    assert scene["target_id"] == 0
    assert len(scene["objects"]) == 2

    prompts = {obj["object_id"]: obj["prompt"] for obj in final_scene["objects"]}
    assert prompts == {0: "light blue bowl", 1: "orange fruit"}
    assert {obj["object_id"] for obj in world_scene["objects"]} == set(prompts)

    manipulated = [obj for obj in final_scene["objects"] if obj["is_manipulated"]]
    assert [obj["prompt"] for obj in manipulated] == ["orange fruit"]
    assert all("urdf_path" in obj for obj in final_scene["objects"])
    assert all(obj["pose"]["position_m"][2] == 0.0 for obj in world_scene["objects"])


def test_proof_fixture_timing_records_completed_model_steps():
    timing = load_json("pipeline_timing.json")

    assert timing["total_s"] > 0
    steps = {step["name"]: step for step in timing["steps"]}
    assert steps["Step 2: SAM3"]["status"] == "completed"
    assert steps["Step 3: SAM3D"]["status"] == "completed"
    assert steps["Step 5: Export Transforms"]["status"] == "completed"
    assert steps["Step 7: OBJ to URDF"]["status"] == "completed"


def test_proof_fixture_does_not_leak_local_paths_or_secrets():
    for path in PROOF_OUTPUTS.glob("*.json"):
        text = path.read_text()
        assert "/home/linhaobai" not in text
        assert "GEMINI_API_KEY" not in text
        assert "sk-" not in text


def test_proof_fixture_preview_image_is_present():
    preview = PROOF_OUTPUTS / "final_scene_visualization.png"

    assert preview.exists()
    assert preview.stat().st_size > 100_000


def test_proof_fixture_capture_media_is_present():
    video = PROOF_OUTPUTS / "input_capture_preview.mp4"
    frame_names = [
        "input_frame_first.png",
        "input_frame_middle.png",
        "input_frame_last.png",
    ]

    assert video.exists()
    assert video.stat().st_size > 100_000
    for frame_name in frame_names:
        frame = PROOF_OUTPUTS / frame_name
        assert frame.exists()
        assert frame.stat().st_size > 100_000
