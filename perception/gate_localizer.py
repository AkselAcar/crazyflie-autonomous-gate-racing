"""Single-view gate localization via reprojection optimization.

Given the 4 image-space corners of a detected gate plus the drone pose,
we solve for the gate state ``[X, Y, Z, yaw, width]`` that minimizes
the reprojection error. This produces a coarse first estimate from a
single frame; ``gate_triangulator`` then refines it across multiple views.
"""

from __future__ import annotations

from math import atan2, cos, pi, sin

import numpy as np
from scipy.optimize import least_squares

from Hardware_Project import config
from Hardware_Project.planning.geometry import (
    angle_diff,
    cam_position_world,
    euler2rotmat,
    normalize_angle,
)


def _make_reprojection_fn(corners_2d: np.ndarray,
                          drone_pos: np.ndarray,
                          roll: float, pitch: float, drone_yaw: float):
    """Build the residuals(params) function for scipy least_squares."""
    R_b2w = euler2rotmat(roll, pitch, drone_yaw)
    R_c2w = R_b2w @ config.R_CAM2BODY
    R_w2c = R_c2w.T
    cam_pos = cam_position_world(drone_pos, roll, pitch, drone_yaw)

    def residuals(params: np.ndarray) -> np.ndarray:
        X, Y, Z, gate_yaw, W = params
        H = config.GATE_OPENING_HEIGHT

        # Gate-local frame: x=depth (thin), y=width, z=height. Corners are
        # ordered TL, TR, BR, BL to match the detector output.
        local_pts = np.array([
            [0,  W / 2,  H / 2],   # TL
            [0, -W / 2,  H / 2],   # TR
            [0, -W / 2, -H / 2],   # BR
            [0,  W / 2, -H / 2],   # BL
        ], dtype=np.float64)

        cg, sg = cos(gate_yaw), sin(gate_yaw)
        R_gate = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        world_pts = (R_gate @ local_pts.T).T + np.array([X, Y, Z])

        res = []
        for i in range(4):
            p_world = world_pts[i]
            p_cam = R_w2c @ (p_world - cam_pos)
            if p_cam[2] <= 0.01:
                # Behind camera or extremely close — push residuals away.
                res.extend([100.0, 100.0])
                continue
            u = config.FOCAL_PX * p_cam[0] / p_cam[2] + config.CAM_CX
            v = config.FOCAL_PX * p_cam[1] / p_cam[2] + config.CAM_CY
            res.append(u - corners_2d[i, 0])
            res.append(v - corners_2d[i, 1])
        return np.array(res)

    return residuals


def localize_gate_reprojection(corners_2d: np.ndarray,
                               drone_pos: np.ndarray,
                               roll: float, pitch: float, drone_yaw: float,
                               initial_guess: list[float],
                               ) -> tuple[np.ndarray, float]:
    """Estimate ``[X, Y, Z, yaw, width]`` from a single view. Returns (params, cost)."""
    fn = _make_reprojection_fn(corners_2d, drone_pos, roll, pitch, drone_yaw)
    bounds_lo = [-3.0, -3.0, 0.3, -np.inf,
                 config.GATE_WIDTH_BOUNDS[0] - 0.05]
    bounds_hi = [3.0, 3.0, 2.5, np.inf,
                 config.GATE_WIDTH_BOUNDS[1] + 0.05]
    result = least_squares(fn, initial_guess,
                           bounds=(bounds_lo, bounds_hi),
                           method="trf", max_nfev=200)
    result.x[3] = normalize_angle(result.x[3])
    return result.x, float(result.cost)


def enforce_correct_yaw_orientation(est_x: float, est_y: float,
                                    est_yaw: float) -> float:
    """Flip yaw by 180° if the optimizer found the gate "viewed from behind".

    The convention is that the gate yaw is its normal pointing roughly
    towards the next gate in CCW order.
    """
    radial_angle = atan2(est_y - config.CIRCLE_CENTRE[1],
                         est_x - config.CIRCLE_CENTRE[0])
    expected_yaw = radial_angle + pi / 2
    diff_normal = abs(angle_diff(expected_yaw, est_yaw))
    diff_flipped = abs(angle_diff(expected_yaw, est_yaw + pi))
    if diff_flipped < diff_normal:
        return normalize_angle(est_yaw + pi)
    return est_yaw


def gate_pixel_width(corners: np.ndarray) -> float:
    """Approximate pixel width (mean of top and bottom edges)."""
    if corners is None:
        return 0.0
    top_w = float(np.linalg.norm(corners[1] - corners[0]))
    bot_w = float(np.linalg.norm(corners[2] - corners[3]))
    return (top_w + bot_w) / 2.0
