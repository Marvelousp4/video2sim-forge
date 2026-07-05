import json

import numpy as np
import pytest

from scripts.validate_hand_world import validate_hand_world


def test_validate_hand_world_accepts_mano_shape(tmp_path):
    path = tmp_path / "hand_world.npy"
    data = np.zeros((3, 21, 3), dtype=np.float32)
    data[:, 0, 0] = [0.0, 0.1, 0.3]
    np.save(path, data)

    summary = validate_hand_world(path)

    assert summary["shape"] == [3, 21, 3]
    assert summary["frames"] == 3
    assert summary["keypoints"] == 21
    assert summary["wrist_step_max"] == pytest.approx(0.2)
    json.dumps(summary)


def test_validate_hand_world_rejects_wrong_shape(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((3, 20, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected shape"):
        validate_hand_world(path)


def test_validate_hand_world_rejects_nan(tmp_path):
    path = tmp_path / "nan.npy"
    data = np.zeros((3, 21, 3), dtype=np.float32)
    data[0, 0, 0] = np.nan
    np.save(path, data)

    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_hand_world(path)
