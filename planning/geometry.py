"""Map geometry and rotation helpers.

Same conventions as the Webots port (CCW sectors of pi/3 around
``CIRCLE_CENTRE``), with constants now sourced from ``config`` so the
real arena (origin (0,0), 5 gates, INNER=0.2m, OUTER=1.5m) is honored.
"""

from __future__ import annotations

from math import atan2, cos, pi, sin

import numpy as np

import config


def normalize_angle(a: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    while a >= pi:
        a -= 2 * pi
    while a < -pi:
        a += 2 * pi
    return a


def angle_diff(target: float, current: float) -> float:
    """Shortest signed difference (target - current), in [-pi, pi)."""
    return normalize_angle(target - current)


def angular_bounds_for_segment(seg_idx: int) -> tuple[float, float]:
    """Sector ``seg_idx`` lower/upper angular bounds in world frame.

    The shift ``config.SECTOR_ANGLE_OFFSET`` aligns sector 0 (the take-off
    sector) with the direction of the take-off pad — change it (not this
    function) when the Lighthouse frame is redefined.
    """
    lo = (2 * seg_idx - 0.5) * config.SEGMENT_ANGULAR_SIZE
    hi = (2 * seg_idx + 0.5) * config.SEGMENT_ANGULAR_SIZE
    shift = config.SECTOR_ANGLE_OFFSET
    return normalize_angle(lo + shift), normalize_angle(hi + shift)


def gate_search_corners(gate_idx: int) -> list[np.ndarray]:
    """Return the 4 (x,y) corners of the search region for gate ``gate_idx``.

    Gate ``i`` lives in segment ``i + 1``. Corners are ordered so that
    consecutive ones share an edge of the polar sector (useful for
    ``cv2.pointPolygonTest``).
    """
    seg = gate_idx + 1
    ang_lo, ang_hi = angular_bounds_for_segment(seg)
    if ang_hi < ang_lo:
        ang_hi += 2 * pi
    pairs = [
        (ang_lo, config.INNER_RADIUS),
        (ang_lo, config.OUTER_RADIUS),
        (ang_hi, config.OUTER_RADIUS),
        (ang_hi, config.INNER_RADIUS),
    ]
    return [
        np.array([
            config.CIRCLE_CENTRE[0] + r * cos(a),
            config.CIRCLE_CENTRE[1] + r * sin(a),
        ])
        for a, r in pairs
    ]


def gate_search_center(gate_idx: int) -> np.ndarray:
    return np.mean(gate_search_corners(gate_idx), axis=0)


def expected_gate_yaw(gate_idx: int) -> float:
    """Expected gate normal (pointing roughly towards the next gate, CCW)."""
    seg = gate_idx + 1
    ang_lo, ang_hi = angular_bounds_for_segment(seg)
    if ang_hi < ang_lo:
        ang_hi += 2 * pi
    return normalize_angle((ang_lo + ang_hi) / 2.0 + pi / 2)


# Rotation helpers
def euler2rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body→inertial rotation matrix (ZYX Euler convention)."""
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return R_yaw @ R_pitch @ R_roll


def cam_position_world(drone_pos: np.ndarray,
                       roll: float, pitch: float, yaw: float) -> np.ndarray:
    return drone_pos + euler2rotmat(roll, pitch, yaw) @ config.CAM_OFFSET_BODY


def R_cam2world(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return euler2rotmat(roll, pitch, yaw) @ config.R_CAM2BODY


def compute_traversal_direction(gate_pos_xy: np.ndarray,
                                gate_yaw: float) -> np.ndarray:
    """Unit 2D vector for flying through the gate in CCW direction."""
    radial = gate_pos_xy[:2] - config.CIRCLE_CENTRE
    radial_unit = radial / (np.linalg.norm(radial) + 1e-12)
    ccw_tangent = np.array([-radial_unit[1], radial_unit[0]])
    gate_normal = np.array([cos(gate_yaw), sin(gate_yaw)])
    if np.dot(gate_normal, ccw_tangent) >= 0:
        return gate_normal
    return -gate_normal
