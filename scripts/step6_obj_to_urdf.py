#!/usr/bin/env python3
"""
Step 6 (STRICT+FALLBACK): Convert OBJ meshes to URDF with robust physics properties.

I/O stays the same as your original:
  --scene_json   scene_output_new.json
  --gemini_json  gemini_scene.json
  --output_json  scene_output_final.json

Key strict logic:
- OBJ parsing supports negative indices.
- Mesh volume via signed tetrahedra on triangulated faces.
- Compute bbox dims + bbox volume.
- If mesh_volume/bbox_volume is too small (thin shell / non-watertight) OR too big, fallback:
    volume = bbox_volume * fill_ratio
  fill_ratio depends on material & simple shape heuristic.
- Mass = density * volume with optional clamp.
- COM uses bbox center, inertia uses solid box about COM.
- URDF mesh path uses os.path.relpath (portable).

You can tune thresholds & fill ratios at top.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional


# -----------------------------
# Tunables (STRICT behavior)
# -----------------------------
VOLUME_RATIO_MIN = 0.10   # if mesh_volume / bbox_volume < this => fallback
VOLUME_RATIO_MAX = 1.20   # if mesh_volume / bbox_volume > this => fallback
MASS_CLAMP_MIN_KG = 1e-4  # 0.1 g
MASS_CLAMP_MAX_KG = 50.0  # 50 kg

# fill ratios used in fallback volume = bbox_volume * fill_ratio
FILL_RATIO_DEFAULT = 0.35
FILL_RATIO_BOXLIKE = 0.75
FILL_RATIO_CYL_LIKE = 0.55

# "thin shell" objects tend to have tiny solid volume; if detected, you probably want bbox-based volume
# (especially for recon meshes)


# -----------------------------
# Material properties database
# -----------------------------
MATERIAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "metal": {
        "density": 7850.0,
        "friction_static": 0.74,
        "friction_dynamic": 0.57,
        "friction_rolling": 0.001,
        "restitution": 0.3,
    },
    "plastic": {
        "density": 1200.0,
        "friction_static": 0.35,
        "friction_dynamic": 0.25,
        "friction_rolling": 0.002,
        "restitution": 0.5,
    },
    "wood": {
        "density": 700.0,
        "friction_static": 0.42,
        "friction_dynamic": 0.35,
        "friction_rolling": 0.003,
        "restitution": 0.4,
    },
    "cardboard": {
        "density": 689.0,
        "friction_static": 0.50,
        "friction_dynamic": 0.40,
        "friction_rolling": 0.004,
        "restitution": 0.2,
    },
    "glass": {
        "density": 2500.0,
        "friction_static": 0.94,
        "friction_dynamic": 0.40,
        "friction_rolling": 0.001,
        "restitution": 0.15,
    },
    "ceramic": {
        "density": 2400.0,
        "friction_static": 0.60,
        "friction_dynamic": 0.45,
        "friction_rolling": 0.002,
        "restitution": 0.25,
    },
    "rubber": {
        "density": 1100.0,
        "friction_static": 0.90,
        "friction_dynamic": 0.80,
        "friction_rolling": 0.005,
        "restitution": 0.8,
    },
    "fabric": {
        "density": 500.0,
        "friction_static": 0.40,
        "friction_dynamic": 0.30,
        "friction_rolling": 0.010,
        "restitution": 0.1,
    },
    "foam": {
        "density": 120.0,  # rough
        "friction_static": 0.60,
        "friction_dynamic": 0.50,
        "friction_rolling": 0.004,
        "restitution": 0.35,
    },
    "unknown": {
        "density": 1000.0,
        "friction_static": 0.50,
        "friction_dynamic": 0.40,
        "friction_rolling": 0.003,
        "restitution": 0.3,
    },
}

MATERIAL_ALIASES = {
    "aluminum": "metal",
    "steel": "metal",
    "tin": "metal",
    "paper": "cardboard",
    "box": "cardboard",
    "carton": "cardboard",
    "porcelain": "ceramic",
    "china": "ceramic",
    "silicone": "rubber",
    "cloth": "fabric",
    "textile": "fabric",
    "sponge": "foam",
    "styrofoam": "foam",
}


# -----------------------------
# OBJ parsing
# -----------------------------
def parse_obj_file(obj_path: str) -> Tuple[np.ndarray, List[List[int]]]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with open(obj_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue

                face: List[int] = []
                for part in parts:
                    token = part.split("/")[0]
                    if not token:
                        continue
                    idx = int(token)
                    if idx < 0:
                        idx = len(vertices) + idx  # relative index
                    else:
                        idx = idx - 1  # 1-based to 0-based
                    face.append(idx)

                if len(face) >= 3 and all(0 <= i < len(vertices) for i in face):
                    faces.append(face)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"OBJ has no vertices/faces: {obj_path}")

    return np.asarray(vertices, dtype=np.float64), faces


def triangulate_faces(faces: List[List[int]]) -> np.ndarray:
    tris: List[Tuple[int, int, int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        v0 = face[0]
        for i in range(1, len(face) - 1):
            tris.append((v0, face[i], face[i + 1]))
    if not tris:
        raise ValueError("No triangles after triangulation.")
    return np.asarray(tris, dtype=np.int64)


# -----------------------------
# Geometry / mass props
# -----------------------------
def bbox_center_and_dims(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    center = 0.5 * (mn + mx)
    dims = mx - mn
    return center, dims


def bbox_volume(dims: np.ndarray) -> float:
    lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
    if lx <= 0 or ly <= 0 or lz <= 0:
        return 0.0
    return lx * ly * lz


def calculate_mesh_volume(vertices: np.ndarray, faces: List[List[int]]) -> float:
    tris = triangulate_faces(faces)
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0))
    return abs(vol)


def inertia_box_about_com(mass: float, dims: np.ndarray) -> np.ndarray:
    lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
    ixx = (mass / 12.0) * (ly * ly + lz * lz)
    iyy = (mass / 12.0) * (lx * lx + lz * lz)
    izz = (mass / 12.0) * (lx * lx + ly * ly)
    return np.array([[ixx, 0.0, 0.0],
                     [0.0, iyy, 0.0],
                     [0.0, 0.0, izz]], dtype=np.float64)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def normalize_material(material_type: Optional[str]) -> str:
    if not material_type:
        return "unknown"
    m = material_type.strip().lower()
    if m in MATERIAL_PROPERTIES:
        return m
    if m in MATERIAL_ALIASES:
        return MATERIAL_ALIASES[m]
    for k in MATERIAL_PROPERTIES.keys():
        if k != "unknown" and (k in m or m in k):
            return k
    for k, v in MATERIAL_ALIASES.items():
        if k in m:
            return v
    return "unknown"


def get_material_properties(material_type: str) -> Dict[str, float]:
    key = normalize_material(material_type)
    return MATERIAL_PROPERTIES.get(key, MATERIAL_PROPERTIES["unknown"])


def choose_fill_ratio(dims: np.ndarray, material_key: str) -> float:
    """
    Very simple shape heuristic:
      - If one dimension is much smaller => thin plate-like (still "boxlike" for volume guess)
      - If two dims are close and third is larger => cylinder-like / can-like
      - Otherwise => default/boxlike
    """
    lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
    d = np.array([lx, ly, lz], dtype=np.float64)
    d_sorted = np.sort(d)
    if d_sorted[2] <= 0:
        return FILL_RATIO_DEFAULT

    # ratios
    thinness = d_sorted[0] / d_sorted[2]  # very small => thin
    a, b, c = d_sorted[0], d_sorted[1], d_sorted[2]

    # can-like: two similar, one different
    two_similar = (abs(b - c) / max(c, 1e-9) < 0.15) or (abs(a - b) / max(b, 1e-9) < 0.15)

    if material_key == "metal":
        return 0.02  
    if material_key == "plastic":
        return 0.06   

    if thinness < 0.2:
        return 0.50  # thin-ish object: avoid going too close to 1.0

    if two_similar:
        return FILL_RATIO_CYL_LIKE

    return FILL_RATIO_BOXLIKE


def robust_volume_with_fallback(vertices: np.ndarray, faces: List[List[int]], material_key: str) -> Tuple[float, Dict[str, float]]:
    center, dims = bbox_center_and_dims(vertices)
    bv = bbox_volume(dims)

    diag = {
        "bbox_lx": float(dims[0]),
        "bbox_ly": float(dims[1]),
        "bbox_lz": float(dims[2]),
        "bbox_volume_m3": float(bv),
    }

    mv = calculate_mesh_volume(vertices, faces)
    diag["mesh_volume_m3_raw"] = float(mv)

    if bv <= 0.0:
        diag["volume_used"] = "mesh"  # no bbox to fallback to
        return mv, diag

    ratio = mv / bv
    diag["mesh_to_bbox_ratio"] = float(ratio)

    if (ratio < VOLUME_RATIO_MIN) or (ratio > VOLUME_RATIO_MAX) or (mv <= 1e-12):
        fill = choose_fill_ratio(dims, material_key)
        diag["fill_ratio"] = float(fill)
        diag["volume_used"] = "bbox_fallback"
        return bv * fill, diag

    diag["volume_used"] = "mesh"
    return mv, diag


# -----------------------------
# URDF writer
# -----------------------------
def create_urdf(
    obj_name: str,
    mesh_abs_path: str,
    mass: float,
    inertia: np.ndarray,
    com_xyz: np.ndarray,
    friction_static: float,
    friction_dynamic: float,
    friction_rolling: float,
    restitution: float,
    output_path: str,
):
    urdf_path = Path(output_path)
    urdf_dir = urdf_path.parent
    mesh_abs = Path(mesh_abs_path).resolve()
    mesh_rel = os.path.relpath(str(mesh_abs), start=str(urdf_dir))

    urdf_content = f"""<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="base_link">
    <inertial>
      <origin xyz="{com_xyz[0]:.9f} {com_xyz[1]:.9f} {com_xyz[2]:.9f}" rpy="0 0 0"/>
      <mass value="{mass:.9f}"/>
      <inertia
        ixx="{inertia[0,0]:.12g}" ixy="{inertia[0,1]:.12g}" ixz="{inertia[0,2]:.12g}"
        iyy="{inertia[1,1]:.12g}" iyz="{inertia[1,2]:.12g}"
        izz="{inertia[2,2]:.12g}"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel}"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel}"/>
      </geometry>
    </collision>
  </link>

  <gazebo reference="base_link">
    <mu1>{friction_static:.6f}</mu1>
    <mu2>{friction_dynamic:.6f}</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <minDepth>0.001</minDepth>
    <maxVel>0.1</maxVel>
    <material>Gazebo/Grey</material>
  </gazebo>

  <!-- reference only: rolling_friction={friction_rolling:.6f}, restitution={restitution:.6f} -->
</robot>
"""
    urdf_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(urdf_content)


# -----------------------------
# Pipeline logic
# -----------------------------
def process_object(obj_data: Dict[str, Any], obj_index: int, gemini_objects: list, output_dir: Path) -> Dict[str, Any]:
    mesh_path = obj_data.get("mesh_path")
    if not mesh_path:
        print(f"WARNING: No mesh_path for object {obj_index}, skipping")
        return obj_data

    mesh_abs = Path(mesh_path)
    if not mesh_abs.is_absolute():
        mesh_abs = (output_dir / mesh_path).resolve()

    if not mesh_abs.exists():
        print(f"WARNING: Mesh file not found: {mesh_abs}, skipping object {obj_index}")
        return obj_data

    material_type = "unknown"
    if obj_index < len(gemini_objects):
        material_type = gemini_objects[obj_index].get("material_type", "unknown")

    mat_key = normalize_material(material_type)
    mat_props = get_material_properties(material_type)

    print(f"  Object {obj_index}: {mesh_abs.name} (material: {material_type} -> {mat_key})")

    try:
        vertices, faces = parse_obj_file(str(mesh_abs))
    except Exception as e:
        print(f"    ERROR: Failed to parse mesh: {e}")
        return obj_data

    center, dims = bbox_center_and_dims(vertices)
    print(f"    Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"    BBox dims (m): lx={dims[0]:.6f}, ly={dims[1]:.6f}, lz={dims[2]:.6f}")

    try:
        volume, diag = robust_volume_with_fallback(vertices, faces, mat_key)
    except Exception as e:
        print(f"    ERROR: Failed to compute volume: {e}")
        return obj_data

    if diag.get("volume_used") == "bbox_fallback":
        ratio = diag.get("mesh_to_bbox_ratio", float("nan"))
        fill = diag.get("fill_ratio", float("nan"))
        print(f"    Volume used: bbox_fallback  fill_ratio={fill:.3f}  (mesh/bbox={ratio:.3f})  "
              f"vol={volume:.9g} m^3 ({volume*1e6:.2f} cm^3)")
    else:
        print(f"    Volume used: mesh  vol={volume:.9g} m^3 ({volume*1e6:.2f} cm^3)")

    def apply_mass_cap(mass: float, material_key: str, dims: np.ndarray) -> float:
        caps = {
            "metal": 1.0,     
            "plastic": 0.5,   
            "foam": 0.2,      
            "cardboard": 0.5,
            "unknown": 1.0,
        }

        cap = caps.get(material_key, 1.0)

        max_dim = float(np.max(dims))
        if max_dim > 0.25: 
            cap *= 2.0

        return float(min(mass, cap))


    density = float(mat_props["density"])
    mass = density * volume
    mass = apply_mass_cap(mass, mat_key, dims)
    mass = clamp(mass, MASS_CLAMP_MIN_KG, MASS_CLAMP_MAX_KG)

    print(f"    Density: {density:.3f} kg/m^3")
    print(f"    Mass: {mass:.9g} kg ({mass*1000.0:.2f} g)")

    inertia = inertia_box_about_com(mass, dims)
    print(f"    Inertia diag: [{inertia[0,0]:.6e}, {inertia[1,1]:.6e}, {inertia[2,2]:.6e}]")

    urdf_dir = output_dir / "urdfs"
    obj_name = f"obj_{obj_index:04d}"
    urdf_path = urdf_dir / f"{obj_name}.urdf"

    create_urdf(
        obj_name=obj_name,
        mesh_abs_path=str(mesh_abs),
        mass=mass,
        inertia=inertia,
        com_xyz=center,
        friction_static=float(mat_props["friction_static"]),
        friction_dynamic=float(mat_props["friction_dynamic"]),
        friction_rolling=float(mat_props["friction_rolling"]),
        restitution=float(mat_props["restitution"]),
        output_path=str(urdf_path),
    )
    print(f"    URDF created: {urdf_path}")

    obj_data_updated = obj_data.copy()
    obj_data_updated["urdf_path"] = str(urdf_path)
    obj_data_updated["mass"] = float(mass)
    obj_data_updated["inertia"] = {
        "ixx": float(inertia[0, 0]),
        "ixy": float(inertia[0, 1]),
        "ixz": float(inertia[0, 2]),
        "iyy": float(inertia[1, 1]),
        "iyz": float(inertia[1, 2]),
        "izz": float(inertia[2, 2]),
    }
    obj_data_updated["friction"] = {
        "static": float(mat_props["friction_static"]),
        "dynamic": float(mat_props["friction_dynamic"]),
        "rolling": float(mat_props["friction_rolling"]),
        "restitution": float(mat_props["restitution"]),
    }
    obj_data_updated["material_type"] = material_type
    obj_data_updated["volume_m3"] = float(volume)

    return obj_data_updated


def main():
    parser = argparse.ArgumentParser(description="Convert OBJ to URDF with physics properties (STRICT+FALLBACK)")
    parser.add_argument("--scene_json", required=True, help="Path to scene_output_new.json")
    parser.add_argument("--gemini_json", required=True, help="Path to gemini_scene.json")
    parser.add_argument("--output_json", required=True, help="Path to output scene_output_final.json")
    args = parser.parse_args()

    print(f"Loading scene data from: {args.scene_json}")
    with open(args.scene_json, "r") as f:
        scene_data = json.load(f)

    print(f"Loading Gemini data from: {args.gemini_json}")
    with open(args.gemini_json, "r") as f:
        gemini_data = json.load(f)
    gemini_objects = gemini_data.get("objects", [])

    output_dir = Path(args.output_json).parent
    objs = scene_data.get("objects", [])
    print(f"\nProcessing {len(objs)} objects...")

    updated = []
    for i, obj_data in enumerate(objs):
        print(f"\nObject {i}:")
        updated.append(process_object(obj_data, i, gemini_objects, output_dir))

    scene_data["objects"] = updated

    print(f"\nSaving final scene data to: {args.output_json}")
    with open(args.output_json, "w") as f:
        json.dump(scene_data, f, indent=2)

    print("\n" + "=" * 60)
    print("OBJ to URDF conversion completed successfully!")
    print("=" * 60)
    print(f"URDFs saved to: {output_dir / 'urdfs'}")
    print(f"Final scene JSON: {args.output_json}")


if __name__ == "__main__":
    main()
