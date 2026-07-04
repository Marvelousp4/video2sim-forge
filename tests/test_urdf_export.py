import importlib.util
from pathlib import Path

import numpy as np


def load_urdf_module():
    module_path = Path("scripts/step6_obj_to_urdf.py")
    spec = importlib.util.spec_from_file_location("step6_obj_to_urdf", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cube_obj(path: Path):
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "v 0 0 1",
                "v 1 0 1",
                "v 1 1 1",
                "v 0 1 1",
                "f 1 2 3 4",
                "f 5 8 7 6",
                "f 1 5 6 2",
                "f 2 6 7 3",
                "f 3 7 8 4",
                "f 4 8 5 1",
            ]
        )
        + "\n"
    )


def test_process_object_writes_urdf_and_positive_physics(tmp_path: Path):
    urdf = load_urdf_module()
    mesh = tmp_path / "cube.obj"
    write_cube_obj(mesh)

    result = urdf.process_object(
        {
            "object_id": 0,
            "prompt": "plastic cube",
            "mesh_path": str(mesh),
            "pose": {"position_m": [0, 0, 0], "orientation_quat_xyzw": [0, 0, 0, 1]},
        },
        obj_index=0,
        gemini_objects=[{"material_type": "plastic"}],
        output_dir=tmp_path,
    )

    assert result["mass"] > 0
    assert result["volume_m3"] > 0
    assert result["friction"]["static"] == urdf.MATERIAL_PROPERTIES["plastic"]["friction_static"]
    assert result["inertia"]["ixx"] > 0
    assert result["inertia"]["iyy"] > 0
    assert result["inertia"]["izz"] > 0
    assert Path(result["urdf_path"]).exists()
    assert "<robot name=\"obj_0000\">" in Path(result["urdf_path"]).read_text()


def test_inertia_box_about_com_is_symmetric_positive():
    urdf = load_urdf_module()

    inertia = urdf.inertia_box_about_com(2.0, np.array([1.0, 2.0, 3.0]))

    assert np.allclose(inertia, inertia.T)
    assert np.all(np.diag(inertia) > 0)

