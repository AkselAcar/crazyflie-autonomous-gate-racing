"""Multi-view DLT triangulation of gate corners.

The approach is: collect a sliding batch of (camera_pose, corners_4x2)
observations spaced by a minimum baseline, then triangulate each of the
4 gate corners independently using the Direct Linear Transform — solve
for the 3D point that minimizes squared distance to all corner-rays.
"""

from __future__ import annotations

from math import atan2

import numpy as np

import config


def pixel_to_cam_ray(u: float, v: float) -> np.ndarray:
    """Pixel (u,v) → ray direction in the camera frame (un-normalized)."""
    return np.array([u - config.CAM_CX, v - config.CAM_CY, config.FOCAL_PX])


def triangulate_point_multi(
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    corner_idx: int,
) -> np.ndarray | None:
    """Multi-view triangulation of a single gate corner.

    ``observations`` is a list of ``(cam_pos_world, R_cam2world,
    corners_4x2)`` from each photo in the batch. ``corner_idx`` selects
    which of the 4 corners (0=TL, 1=TR, 2=BR, 3=BL) to triangulate.

    Solves the linear system  Σᵢ (I − rᵢ rᵢᵀ) X = Σᵢ (I − rᵢ rᵢᵀ) cᵢ
    where rᵢ is the unit ray from camera ``i`` and cᵢ its position.
    Returns ``None`` if the system is singular (collinear cameras).
    """
    M = np.zeros((3, 3))
    b_vec = np.zeros(3)
    for cam_pos, R_c2w, corners in observations:
        u, v = corners[corner_idx]
        ray_world = R_c2w @ pixel_to_cam_ray(u, v)
        ray_world = ray_world / (np.linalg.norm(ray_world) + 1e-12)
        A_i = np.eye(3) - np.outer(ray_world, ray_world)
        M += A_i
        b_vec += A_i @ cam_pos
    try:
        return np.linalg.solve(M, b_vec)
    except np.linalg.LinAlgError:
        return None


def gate_params_from_corners(world_corners: np.ndarray,
                             ) -> tuple[np.ndarray, float, float]:
    """Extract gate (centre, yaw, width) from triangulated 3D corners.

    Width direction is taken as the horizontal (xy) component of the
    averaged top and bottom edges. Yaw is the gate normal, with the
    same convention as ``enforce_correct_yaw_orientation``.
    """
    center = np.mean(world_corners, axis=0)
    top_vec = world_corners[1] - world_corners[0]   # TL → TR
    bot_vec = world_corners[2] - world_corners[3]   # BL → BR
    width_vec = (top_vec + bot_vec) / 2.0
    width = float(np.linalg.norm(width_vec[:2]))
    gate_yaw = atan2(width_vec[0], -width_vec[1])
    return center, gate_yaw, width
