# V2S2R Hand Trajectory Bridge

This repository exports simulation-ready scene assets. V2S2R contains the
downstream hand trajectory, robot retargeting, and motion-tracking visualization
code. The safest integration is to keep V2S2R as an external checkout and pass
approved hand trajectory and scene outputs between the two repositories.

## What This Adds

Use this bridge when you want to show the next stage after scene reconstruction:

1. Recover or provide MANO right-hand keypoints in world or robot-base frame.
2. Validate the trajectory shape locally.
3. Run V2S2R retargeting on Ubuntu.
4. Visualize wrist, palm, and fingertip target frames in PyBullet.
5. Export screenshots or an MP4 for demos and documentation.

The main hand input expected by V2S2R retargeting is:

```text
hand_world.npy  # shape: (T, 21, 3), MANO right-hand keypoints in world frame
```

## Recommended Ubuntu Layout

```bash
mkdir -p ~/robotics
cd ~/robotics

git clone git@github.com:Marvelousp4/video2sim-forge.git
git clone https://github.com/video2sim2real/video2sim2real.git
```

Keep generated files out of source control:

```bash
mkdir -p ~/robotics/runs/hand_trajectory_demo
```

## Environment

Create a separate conda environment for V2S2R retargeting:

```bash
conda create -n v2s2r-retarget python=3.10 -y
conda activate v2s2r-retarget

cd ~/robotics/video2sim-forge
python -m pip install -U pip
python -m pip install -r requirements-retargeting.txt

cd ~/robotics/video2sim2real
python -m pip install -e .
```

If editable install fails because the upstream repository does not expose a
package build file, run V2S2R commands from the repository root and set:

```bash
export PYTHONPATH=~/robotics/video2sim2real:$PYTHONPATH
```

## Inputs To Prepare

For the scene path, use the normal Video2Sim Forge outputs:

```text
output/<run_name>/
├── scene_output_final.json
├── sam3d_results.json
├── urdfs/
└── final_scene_visualization.png
```

For the hand path, prepare:

```text
~/robotics/runs/hand_trajectory_demo/hand_world.npy
```

The `.npy` file must contain MANO right-hand keypoints with shape `(T, 21, 3)`.
Frame units and coordinate frame must match the robot/world frame expected by
the V2S2R retargeting script you run.

## Validate The Hand Trajectory

```bash
cd ~/robotics/video2sim-forge
python scripts/validate_hand_world.py \
  --hand-world-npy ~/robotics/runs/hand_trajectory_demo/hand_world.npy \
  --summary-json ~/robotics/runs/hand_trajectory_demo/hand_world_summary.json
```

The validator checks:

- array shape is `(T, 21, 3)`
- at least two frames exist
- no NaN or Inf values are present
- wrist motion statistics and XYZ bounds look reasonable

## Run V2S2R Retargeting Visualization

From the V2S2R repository root:

```bash
cd ~/robotics/video2sim2real
conda activate v2s2r-retarget
export PYTHONPATH=$PWD:$PYTHONPATH

python -m retargeting_kinova.retarget_ability \
  --cfg.hand-world-npy ~/robotics/runs/hand_trajectory_demo/hand_world.npy \
  --cfg.visualize
```

Expected output next to the input trajectory:

```text
~/robotics/runs/hand_trajectory_demo/retarget_kinova_ability.json
```

Use `retarget_leap.py` or `retarget_allegro.py` instead when the downstream
robot hand target is LEAP or Allegro.

## Capture Demo Frames

The simple `retarget_ability.py` path opens a PyBullet GUI and draws wrist and
fingertip frames. For saved screenshots, use the V2S2R IK scripts that call
`save_screenshot`, such as:

```bash
python -m retargeting_kinova.gen_motion_disturbance_traj_leap \
  --cfg.visualize
```

That script requires the extra grasp/contact JSON inputs used by V2S2R. Once it
writes frame images, create a video with:

```bash
python utils/create_video.py \
  ~/robotics/runs/hand_trajectory_demo/frames \
  --out ~/robotics/runs/hand_trajectory_demo/hand_trajectory.mp4 \
  --fps 30
```

## What Codex Should Produce Next

Ask Codex to generate or update these items as your Ubuntu run becomes concrete:

- a run-specific config file under `examples/<run_name>/config.yaml`
- a sanitized `examples/<run_name>/README.md` with exact commands and hardware
- a checked-in `hand_world_summary.json` if the data is redistributable
- a small approved preview image or MP4 if the source capture can be public
- a wrapper command that calls V2S2R retargeting from this repository
- tests for any pure-Python format conversion added to the bridge

Do not commit private captures, camera serial numbers, robot-site data, API
keys, or model checkpoints.
