"""Hardware project configuration constants.

Single source of truth for map geometry, camera intrinsics, control
parameters and FSM tuning. Keeping everything here means the perception
and planning modules stay environment-agnostic (the same FSM can in
principle target sim or hardware just by swapping this file).
"""

from __future__ import annotations

from math import pi, tan

import numpy as np

# Crazyflie radio + AI Deck network
CF_URI: str = "radio://0/125/2M/E7E7E7E710"
AI_DECK_IP: str = "192.168.4.1"
AI_DECK_PORT: int = 5000

# Map geometry
# Lighthouse origin is at the centre of the arena. CCW angular sectors of
# 60° (= pi/6 half-width). Same shape as Webots, but the "circle" is now
# centred on (0, 0) instead of (4, 4).
CIRCLE_CENTRE: np.ndarray = np.array([-0.10, 0.0])
INNER_RADIUS: float = 0.2
OUTER_RADIUS: float = 1.5
NUM_GATES: int = 5
NUM_SEGMENTS: int = NUM_GATES + 1
SEGMENT_ANGULAR_SIZE: float = pi / NUM_SEGMENTS  # = pi/6

# Angular offset (rad) that aligns sector 0 (the take-off sector) with the
# direction of TAKE_OFF_POS as seen from CIRCLE_CENTRE. With the
# Lighthouse frame currently in use (take-off at (1, 0), centre at
# (0, 0)), the take-off direction is +x → angle 0, so SECTOR_ANGLE_OFFSET
# is 0. In Webots the take-off was at (1, 4) relative to centre (4, 4) →
# direction -x → offset was pi. If the Lighthouse frame is ever
# redefined, the only constant to change is this one.
SECTOR_ANGLE_OFFSET: float = 0.0

# Map bounds (used for sanity checks on planned waypoints)
X_BOUNDS: tuple[float, float] = (-2.0, 2.0)
Y_BOUNDS: tuple[float, float] = (-1.5, 1.5)
Z_BOUNDS: tuple[float, float] = (0.05, 2.0)

# Take-off and exploration
TAKE_OFF_POS: np.ndarray = np.array([1.2, 0.0, 0.05])
TAKE_OFF_YAW: float = pi  # facing -x, towards the centre of the arena
EXPLORE_ALT: float = 1.2

# Gate physical properties — variable, same as Webots assignment.
# Width is unknown a priori and stays a free parameter in reprojection.
# Height is FIXED in the reprojection model, which means any camera tilt
# or LED-frame-vs-opening mismatch shows up as a biased gate.z estimate
# (not in xy, since W is free and absorbs horizontal mismatches). We
# therefore never use the reprojected gate.z as a control target — we
# use the drone's current z plus the visual servo's vertical correction
# instead. See fsm.py _do_approach / _do_align_traverse.
GATE_OPENING_HEIGHT: float = 0.4
GATE_WIDTH_BOUNDS: tuple[float, float] = (0.3, 0.5)

# AI Deck camera (HM01B0, no calibration — nominal values)
# Resolution and FOV come from the AI Deck 1.1 datasheet. Refine after a
# first calibration pass on a known target.
CAM_WIDTH: int = 324
CAM_HEIGHT: int = 244
CAM_FOV_H: float = pi * 87.0 / 180.0  # ~87° horizontal FOV
FOCAL_PX: float = (CAM_WIDTH / 2.0) / tan(CAM_FOV_H / 2.0)  # ~170 px
CAM_CX: float = CAM_WIDTH / 2.0
CAM_CY: float = CAM_HEIGHT / 2.0

# Camera mounted UNDER the drone, pointing forward (verified from the
# captured image: text reads left-to-right and gates appear at horizon
# level, so the camera is forward-looking and in normal orientation).
# Body frame: x=forward, y=left, z=up.
CAM_OFFSET_BODY: np.ndarray = np.array([0.03, 0.0, -0.025])

# Camera frame: x=right (image), y=down (image), z=forward (out of lens).
# Same orientation as Webots — the only thing that changed is the body
# offset above, not the camera optical axis.
R_CAM2BODY: np.ndarray = np.array([
    [0,  0,  1],   # x_body (fwd)  =  z_cam
    [-1, 0,  0],   # y_body (left) = -x_cam
    [0, -1,  0],   # z_body (up)   = -y_cam
], dtype=np.float64)

# Detection thresholds (LED-based, will be tuned on real images).
# LEDs in the penumbra produce nearly-saturated pixels (>200) so a high
# threshold rejects most of the lab background while keeping the full
# gate ring.
LED_BRIGHTNESS_THRESHOLD: int = 200       # 0–255 grayscale
DET_MIN_AREA: int = 200                   # px², discards tiny reflections
DET_BORDER_MARGIN: int = 3                # reject contours touching the border

# Approach + reprojection-refinement tuning
APPROACH_DIST: float = 0.8                # m, smaller arena → closer approach
# Distance past the gate plane that we command as the traverse target.
# HLC's firmware min-snap reaches zero velocity AT the endpoint, so we
# need to aim past the gate to keep non-zero velocity at the plane
# itself. 0.30 m gives a clean fly-through at HLC_TRAVERSE_DURATION=1.2 s
# over the ~1.1 m total trajectory from align to overshoot.
TRAVERSE_OVERSHOOT: float = 0.30          # m past the gate plane
APPROACH_REFINE_ALPHA: float = 0.12       # blend factor per refinement update
CONVERGENCE_WINDOW: int = 8
CONVERGENCE_MIN_SAMPLES: int = 6
CONVERGENCE_STD_THRESHOLD: float = 0.04
LOST_GATE_TIMEOUT_S: float = 3.0

# Sanity rejection thresholds for per-frame reprojection refinement.
# Each new frame re-runs single-view reprojection from the current drone
# pose; the YAW_SCAN result is a strong prior so any update diverging by
# more than these thresholds is treated as an outlier and dropped.
REFINE_REJECT_CENTER_DRIFT: float = 0.30   # m  (xy distance from current estimate)
REFINE_REJECT_YAW_DRIFT: float = 0.5       # rad (~30°)
REFINE_REJECT_WIDTH_DRIFT: float = 0.20    # m

# Visual-servo centering controller (APPROACH state).
# The controller equalizes left/right and top/bottom gate edge pixel lengths.
# Errors are normalized by the average of the two edges (dimensionless, ±1 max)
# so the gains are in metres of target offset per unit relative imbalance.
#
# Sign convention (camera x=right, y=down, body y=left, body z=up):
#   e_h = (l_left − l_right) / avg  > 0 → drone is LEFT  of gate → target moves RIGHT
#   e_v = (l_top  − l_bottom) / avg > 0 → drone is ABOVE gate    → target moves DOWN
#
# Start with small gains and increase until the drone centres reliably
# without oscillating.  Max correction is clamped to VISUAL_SERVO_MAX_OFFSET.
VISUAL_SERVO_K_H: float = 0.4        # m per unit relative lateral edge imbalance
# K_V is now a dimensionless feedback gain on the metric vertical error:
#   Δz = −K_V × (pixel_err / FOCAL_PX) × dist_to_gate   [metres]
# Unlike the old edge-asymmetry gain this is pitch-compensated and
# distance-aware. Start at 1.0 and tune up if the drone under-corrects.
VISUAL_SERVO_K_V: float = 1.0
# Fallback gain (edge asymmetry) used only when the gate is very close
# (p_cam[2] < 0.1 m) — same unit as the old K_V (m / dimensionless).
VISUAL_SERVO_K_V_FALLBACK: float = 0.5
VISUAL_SERVO_MAX_OFFSET: float = 0.4  # m — hard clamp on each axis

# High-Level Commander durations (seconds). Tune per phase to control speed.
HLC_TAKEOFF_DURATION: float = 2.0
HLC_LAND_DURATION: float = 2.5
HLC_YAW_SCAN_DURATION: float = 3.0      # full sweep across the yaw range
HLC_APPROACH_TICK_DURATION: float = 1.5  # per re-issued go_to during triangulation
HLC_TRAVERSE_DURATION: float = 1.2

# Logging
LOG_DIR: str = "logs"
PLOT_DIR: str = "camera_plots"






# ===================================================
#                       LAP 2&3
# ===================================================
GATE_POS_CSV_PATH = "racing/gate_pos_csv/gates_info_circle_test.csv"
MOTION_PLANNER_VERBOSE = False

VEL_LIM = 25.0
ACC_LIM = 250.0 
PTS_PER_METER = 30.0                # Number of points per meter for trajectory
MIN_T_F = 10.0                      # Minimum trajectory duration [s]
TARGET_AVG_SPEED = 0.5              # Target averaged speed over trajectory [m/s]

ADD_POINT_DIST_FROM_GATE = 0.15                # Distance of points before and after the gate [m]

STARTING_POINT = TAKE_OFF_POS # (0.7, 0.0, 0.75)   # Starting point (x, y, z)
FINISH_POINT_DIST = 0.5             # Finish point distance from last gate [m]

BASE_LOOKAHEAD = 0.0                   # Base lookahead distance for Pure Pursuit [m]
LOOKAHEAD_SPEED_MULTIPLIER = 0.0       # Multiplier for dynamic lookahead based on speed [m/s]