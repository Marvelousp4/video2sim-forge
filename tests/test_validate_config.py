import importlib.util
from pathlib import Path


def load_validator_module():
    module_path = Path("scripts/validate_config.py")
    spec = importlib.util.spec_from_file_location("validate_config", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_minimal_capture(root: Path):
    (root / "depth").mkdir(parents=True)
    (root / "scene_capture" / "image").mkdir(parents=True)
    (root / "scene_capture" / "depth").mkdir(parents=True)
    (root / "color_video.mp4").write_bytes(b"video")
    (root / "depth" / "0.png").write_bytes(b"depth")
    (root / "cam_K.txt").write_text("385.4 385.4 317.4 244.0\n")
    (root / "scene_capture" / "image" / "0.png").write_bytes(b"image")
    (root / "scene_capture" / "depth" / "0.png").write_bytes(b"depth")


def test_validate_config_passes_for_skip_heavy_local_config(tmp_path: Path, monkeypatch):
    validator = load_validator_module()
    capture = tmp_path / "capture"
    write_minimal_capture(capture)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
input_dir: "{capture}"
output_dir: "{tmp_path / 'out'}"
skip_gemini: true
skip_sam3: true
skip_sam3d: true
skip_export: true
skip_obj_to_urdf: true
"""
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks = validator.validate_config(config)

    assert not [check for check in checks if check.status == "ERROR"]
    assert any(check.name == "PyTorch CUDA" and check.status == "SKIP" for check in checks)


def test_validate_config_reports_missing_gemini_key(tmp_path: Path, monkeypatch):
    validator = load_validator_module()
    capture = tmp_path / "capture"
    write_minimal_capture(capture)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
input_dir: "{capture}"
output_dir: "{tmp_path / 'out'}"
skip_gemini: false
skip_sam3: true
skip_sam3d: true
skip_export: true
"""
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks = validator.validate_config(config)

    assert any(
        check.status == "ERROR" and check.name == "GEMINI_API_KEY"
        for check in checks
    )

