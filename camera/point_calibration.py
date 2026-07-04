import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.font_manager as fm
fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))

plt.rcParams['font.family'] = 'Nimbus Roman'

# Compute optimal rigid transformation (R, t)
def compute_rigid_transform(A, B):
    """ Compute optimal rigid transformation matrix (R, t) so that B ≈ R * A + t """
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)

    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    t_opt = centroid_B - R_opt @ centroid_A
    return R_opt, t_opt

# Convert quaternion to Euler angles (3D rotation vector)
def quat_to_euler(quaternion):
    """ Convert quaternion to 3D rotation vector (Euler angles) """
    return R.from_quat(quaternion).as_euler('xyz', degrees=False)

# Draw coordinate axes
def draw_axes(ax, origin, rotation_matrix, scale=0.05, label=""):
    """
    Draw 3D coordinate axes at a given origin with a specified rotation.
    """
    x_axis = origin + scale * rotation_matrix[:, 0]
    y_axis = origin + scale * rotation_matrix[:, 1]
    z_axis = origin + scale * rotation_matrix[:, 2]

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r', linewidth=2)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g', linewidth=2)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b', linewidth=2)
    ax.text(origin[0], origin[1], origin[2], label, color='black')