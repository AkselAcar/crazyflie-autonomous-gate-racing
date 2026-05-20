"""Lap-1 FSM, ported from the Webots assignment to High-Level Commander.

Pipeline per gate:
    TAKEOFF → YAW_SCAN → APPROACH (multi-view triangulation)
            → ALIGN_AND_TRAVERSE → (next gate or LAND)
With FALLBACK_SCAN as a safety net when the gate is lost.

Key adaptations vs the Webots original:
- No CommandSmoother. The High-Level Commander generates a min-snap
  trajectory in firmware, so we issue ``cf.go_to(...)`` whenever the
  target meaningfully changes and let the drone fly there.
- ``tick`` is event-driven — it doesn't return a setpoint each tick.
  It computes a desired (x, y, z, yaw) target and re-issues ``go_to``
  only when the target moved by ``REISSUE_*`` or after
  ``MIN_REISSUE_INTERVAL`` has elapsed.
- A LAND state is added at the end (no RACE phase: lap 1 only).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from math import atan2, cos, sin

import cv2
import numpy as np

import config
from hardware.crazyflie_link import CrazyflieLink, DroneState
from perception.gate_detector import detect_all_gate_contours
from perception.gate_localizer import (
    enforce_correct_yaw_orientation,
    gate_pixel_width,
    localize_gate_reprojection,
)
from planning.geometry import (
    R_cam2world,
    angle_diff,
    angular_bounds_for_segment,
    cam_position_world,
    compute_traversal_direction,
    expected_gate_yaw,
    gate_search_center,
    gate_search_corners,
    normalize_angle,
)

log = logging.getLogger(__name__)


@dataclass
class GateEstimate:
    position: np.ndarray  # shape (3,) — world frame
    yaw: float            # gate normal, CCW convention
    width: float          # m


class GateFSM:
    """Finite-state machine that flies lap 1 and lands."""

    # States
    TAKEOFF = 0
    YAW_SCAN = 1
    APPROACH = 2
    ALIGN_AND_TRAVERSE = 3
    FALLBACK_SCAN = 4
    LAND = 5
    DONE = 6

    _STATE_NAMES = {
        TAKEOFF: "TAKEOFF", YAW_SCAN: "YAW_SCAN", APPROACH: "APPROACH",
        ALIGN_AND_TRAVERSE: "ALIGN_AND_TRAVERSE",
        FALLBACK_SCAN: "FALLBACK_SCAN", LAND: "LAND", DONE: "DONE",
    }

    # HLC re-issue thresholds (avoid spamming the radio link).
    # MIN_REISSUE_INTERVAL = 0.1s ≈ 10 Hz, matched to the typical AI Deck
    # JPEG-over-WiFi rate. Lower wouldn't help (we'd re-issue go_to on the
    # same image); higher (the previous 0.4 s) bottlenecked the visual
    # servo to 2.5 Hz — 20 cm of approach travel between corrections at
    # NOMINAL_SPEED.
    REISSUE_POS_THRESHOLD = 0.05   # m
    REISSUE_YAW_THRESHOLD = 0.05   # rad
    MIN_REISSUE_INTERVAL = 0.1     # s

    # Arrival tolerances
    ARRIVAL_POS_TOL = 0.10
    ARRIVAL_Z_TOL = 0.08
    ARRIVAL_YAW_TOL = 0.08
    ARRIVAL_VEL_TOL = 0.10

    # Time budget per metre / per radian for sizing HLC durations.
    NOMINAL_SPEED = 0.5      # m/s
    YAW_RATE = 0.6           # rad/s
    MIN_SEG_DURATION = 0.6   # s — floor on any go_to duration

    # YAW_SCAN runs slower than the general yaw rate so the camera frames
    # are not motion-blurred and the detector has more shots at each gate.
    SCAN_YAW_RATE = 0.30     # rad/s (~17°/s)
    SCAN_MIN_DURATION = 2.0  # s — floor on a single yaw-scan leg

    # Detection / triangulation tuning (reuses config)
    MIN_GATE_PX_WIDTH = 18

    def __init__(self, cf: CrazyflieLink) -> None:
        self.cf = cf
        self.state: int = self.TAKEOFF
        self.current_gate: int = 0
        self.gates: list[GateEstimate | None] = [None] * config.NUM_GATES

        # HLC bookkeeping
        self._cmd_target: np.ndarray | None = None  # [x, y, z, yaw]
        self._cmd_t: float = 0.0
        self._takeoff_issued: bool = False
        self._land_issued: bool = False

        # Yaw scan state
        self._scan_yaw_target: float = 0.0
        self._scan_second_target: float = 0.0
        self._scan_phase: int = 0
        self._scan_best_detection: np.ndarray | None = None
        self._scan_best_cost: float = float("inf")
        self._scan_anchor_xy: np.ndarray | None = None  # hover xy during scan

        # Approach state — single-view reprojection only, no multi-view
        # triangulation (which was unstable when the drone moves along
        # the gate normal: parallel rays → poor depth estimate).
        self._approach_pos_history: list[np.ndarray] = []
        self._approach_last_seen_t: float | None = None
        # Persistent z setpoint accumulates servo corrections without
        # re-anchoring to drone_pos[2] each tick (prevents early z commit).
        self._servo_z_setpoint: float | None = None
        self._servo_z_settled: int = 0  # consecutive frames with small z error
        # Persistent lateral (xy) servo offset — the horizontal analogue of
        # _servo_z_setpoint. Accumulated during APPROACH/ALIGN and applied to
        # BOTH the align point AND the traverse target, so the traverse flies
        # straight along the corrected gate axis instead of a diagonal through
        # the un-corrected reprojected centre (which clipped a gate leg).
        self._servo_xy_offset: np.ndarray | None = None
        self._servo_xy_settled: int = 0  # consecutive frames with small xy error

        # Traverse state
        self._traverse_target: np.ndarray | None = None
        self._traverse_started: bool = False

        # Fallback scan state — discrete yaw_scans at 3 viewpoints on the
        # radial at ang_hi of the previous sector. See _init_fallback_scan.
        self._fallback_positions: list[np.ndarray] = []
        self._fallback_pos_idx: int = 0
        self._fallback_substate: str = "MOVING"  # "MOVING" | "SCANNING"
        self._fallback_look_yaw: float | None = None  # locked during MOVING

        # Landing state. _land_return_xy is set from the resting pose at
        # take-off (see _do_takeoff), so the drone lands where it took off.
        self._land_phase: int = 0
        self._land_return_xy: np.ndarray = np.zeros(2)

    # Public API
    def tick(self, sd: DroneState, image: np.ndarray | None) -> bool:
        """Drive the FSM one step. Returns False when mission complete."""
        if self.state == self.DONE:
            return False

        drone_pos = np.array([sd.x_global, sd.y_global, sd.z_global])

        if self.state == self.TAKEOFF:
            self._do_takeoff(sd, drone_pos)
        elif self.state == self.YAW_SCAN:
            self._do_yaw_scan(sd, image, drone_pos)
        elif self.state == self.APPROACH:
            self._do_approach(sd, image, drone_pos)
        elif self.state == self.ALIGN_AND_TRAVERSE:
            self._do_align_traverse(sd, image, drone_pos)
        elif self.state == self.FALLBACK_SCAN:
            self._do_fallback_scan(sd, image, drone_pos)
        elif self.state == self.LAND:
            self._do_land(sd, drone_pos)

        return self.state != self.DONE

    # State transitions (logged)
    def _set_state(self, new_state: int) -> None:
        if new_state != self.state:
            log.info("FSM: %s -> %s",
                     self._STATE_NAMES[self.state],
                     self._STATE_NAMES[new_state])
            self.state = new_state

    # HLC helpers
    def _reissue_needed(self, target: np.ndarray) -> bool:
        if self._cmd_target is None:
            return True
        if time.monotonic() - self._cmd_t < self.MIN_REISSUE_INTERVAL:
            return False
        pos_moved = np.linalg.norm(target[:3] - self._cmd_target[:3])
        yaw_moved = abs(angle_diff(target[3], self._cmd_target[3]))
        return (pos_moved > self.REISSUE_POS_THRESHOLD
                or yaw_moved > self.REISSUE_YAW_THRESHOLD)

    def _issue_goto(self, x: float, y: float, z: float, yaw: float,
                    duration: float | None = None,
                    cur_pos: np.ndarray | None = None,
                    cur_yaw: float | None = None) -> None:
        target = np.array([x, y, z, yaw])
        if not self._reissue_needed(target):
            return
        if duration is None:
            if cur_pos is None or cur_yaw is None:
                duration = self.MIN_SEG_DURATION
            else:
                dist = np.linalg.norm(target[:3] - cur_pos)
                yaw_d = abs(angle_diff(yaw, cur_yaw))
                duration = max(self.MIN_SEG_DURATION,
                               dist / self.NOMINAL_SPEED,
                               yaw_d / self.YAW_RATE)
        self.cf.go_to(x, y, z, yaw, duration)
        self._cmd_target = target
        self._cmd_t = time.monotonic()

    def _at_target(self, sd: DroneState, target_xy: np.ndarray | None = None,
                   target_z: float | None = None,
                   target_yaw: float | None = None,
                   pos_tol: float | None = None,
                   yaw_tol: float | None = None) -> bool:
        pos_tol = pos_tol if pos_tol is not None else self.ARRIVAL_POS_TOL
        yaw_tol = yaw_tol if yaw_tol is not None else self.ARRIVAL_YAW_TOL
        if target_xy is not None:
            if np.linalg.norm(np.array([sd.x_global, sd.y_global]) - target_xy) > pos_tol:
                return False
        if target_z is not None:
            if abs(sd.z_global - target_z) > self.ARRIVAL_Z_TOL:
                return False
        if target_yaw is not None:
            if abs(angle_diff(target_yaw, sd.yaw)) > yaw_tol:
                return False
        # Velocity check — make sure the drone has settled.
        speed = (sd.v_x ** 2 + sd.v_y ** 2 + sd.v_z ** 2) ** 0.5
        return speed < self.ARRIVAL_VEL_TOL

    # TAKEOFF
    def _do_takeoff(self, sd: DroneState, drone_pos: np.ndarray) -> None:
        if not self._takeoff_issued:
            log.info("Resting pose: x=%.2f y=%.2f z=%.2f yaw=%.2f",
                     sd.x_global, sd.y_global, sd.z_global, sd.yaw)
            self._land_return_xy = np.array([sd.x_global, sd.y_global])
            self.cf.takeoff(config.EXPLORE_ALT, config.HLC_TAKEOFF_DURATION)
            self._takeoff_issued = True
            self._cmd_t = time.monotonic()
            return
        if time.monotonic() - self._cmd_t < config.HLC_TAKEOFF_DURATION:
            return
        if abs(sd.z_global - config.EXPLORE_ALT) < self.ARRIVAL_Z_TOL:
            self._init_yaw_scan(self.current_gate, drone_pos, sd.yaw)
            self._set_state(self.YAW_SCAN)

    # YAW SCAN
    def _init_yaw_scan(self, gate_idx: int, drone_pos: np.ndarray,
                       drone_yaw: float) -> None:
        corners = gate_search_corners(gate_idx)
        rel_yaws = [
            angle_diff(atan2(c[1] - drone_pos[1], c[0] - drone_pos[0]),
                       drone_yaw)
            for c in corners
        ]
        margin = 0.12  # ~7° margin around extremes
        right_limit = normalize_angle(drone_yaw + min(rel_yaws) - margin)
        left_limit = normalize_angle(drone_yaw + max(rel_yaws) + margin)

        # Start with whichever extreme is closer (less time to reach).
        if abs(angle_diff(right_limit, drone_yaw)) <= abs(angle_diff(left_limit, drone_yaw)):
            self._scan_yaw_target = right_limit
            self._scan_second_target = left_limit
        else:
            self._scan_yaw_target = left_limit
            self._scan_second_target = right_limit

        self._scan_phase = 0
        self._scan_best_detection = None
        self._scan_best_cost = float("inf")
        self._scan_anchor_xy = drone_pos[:2].copy()
        self._cmd_target = None  # force re-issue at the new target

    def _scan_duration(self, current_yaw: float, target_yaw: float) -> float:
        """Slow yaw rotation during scanning to limit motion blur."""
        return max(self.SCAN_MIN_DURATION,
                   abs(angle_diff(target_yaw, current_yaw)) / self.SCAN_YAW_RATE)

    def _yaw_scan_step(self, sd: DroneState, image: np.ndarray | None,
                       drone_pos: np.ndarray) -> str:
        """One tick of the yaw-sweep search.

        Hovers at ``_scan_anchor_xy`` and rotates between
        ``_scan_yaw_target`` and ``_scan_second_target`` while running
        the detector on every frame. Used by both the regular YAW_SCAN
        state and the FALLBACK_SCAN sub-state, so transitions stay in
        the caller and this helper only reports progress.

        Returns:
            ``"running"``       — still sweeping, keep ticking.
            ``"found"``         — sweep complete, a valid detection is
                                  stored in ``_scan_best_detection``.
            ``"no_detection"``  — sweep complete with nothing detected.
        """
        if image is not None:
            self._try_detect_gate_during_scan(image, drone_pos,
                                              sd.roll, sd.pitch, sd.yaw)

        anchor = self._scan_anchor_xy
        z = config.EXPLORE_ALT

        if self._scan_phase == 0:
            duration = self._scan_duration(sd.yaw, self._scan_yaw_target)
            self._issue_goto(anchor[0], anchor[1], z, self._scan_yaw_target,
                             duration=duration,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_yaw=self._scan_yaw_target):
                self._scan_phase = 1
            return "running"
        if self._scan_phase == 1:
            duration = self._scan_duration(sd.yaw, self._scan_second_target)
            self._issue_goto(anchor[0], anchor[1], z, self._scan_second_target,
                             duration=duration,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_yaw=self._scan_second_target):
                self._scan_phase = 2
            return "running"
        return "found" if self._scan_best_detection is not None else "no_detection"

    def _do_yaw_scan(self, sd: DroneState, image: np.ndarray | None,
                     drone_pos: np.ndarray) -> None:
        result = self._yaw_scan_step(sd, image, drone_pos)
        if result == "running":
            return
        if result == "found":
            p = self._scan_best_detection
            self.gates[self.current_gate] = GateEstimate(
                position=np.array([p[0], p[1], p[2]]),
                yaw=p[3], width=p[4],
            )
            log.info("Gate %d found at %s yaw=%.2f w=%.2f",
                     self.current_gate,
                     np.round(self.gates[self.current_gate].position, 2),
                     self.gates[self.current_gate].yaw,
                     self.gates[self.current_gate].width)
            self._init_approach()
            self._set_state(self.APPROACH)
        else:
            log.warning("No detection during yaw scan for gate %d",
                        self.current_gate)
            self._init_fallback_scan(self.current_gate)
            self._set_state(self.FALLBACK_SCAN)

    def _try_detect_gate_during_scan(self, image: np.ndarray,
                                     drone_pos: np.ndarray,
                                     roll: float, pitch: float,
                                     yaw: float) -> None:
        detections = detect_all_gate_contours(image)
        if not detections:
            return
        center = gate_search_center(self.current_gate)
        exp_yaw = expected_gate_yaw(self.current_gate)
        initial = [center[0], center[1], config.EXPLORE_ALT,
                   exp_yaw, config.GATE_OPENING_HEIGHT]

        search_poly = np.array(gate_search_corners(self.current_gate),
                               dtype=np.float32)

        for corners, _area, _idx in detections:
            params, cost = localize_gate_reprojection(
                corners, drone_pos, roll, pitch, yaw, initial_guess=initial,
            )
            if params[2] < 0.3 or params[2] > 2.5:
                continue
            est_pos = (float(params[0]), float(params[1]))
            dist_to_region = cv2.pointPolygonTest(search_poly, est_pos, True)
            if dist_to_region < -0.2:
                continue
            params[3] = enforce_correct_yaw_orientation(
                params[0], params[1], params[3])
            # Compensate the uncalibrated camera vertical bias: reprojection
            # places the gate ~GATE_Z_OFFSET too low in z. xy/yaw/width are
            # unaffected by the bias. See config.GATE_Z_OFFSET.
            params[2] += config.GATE_Z_OFFSET
            if cost < self._scan_best_cost:
                self._scan_best_cost = cost
                self._scan_best_detection = params

    # APPROACH (single-view reprojection refinement)
    def _init_approach(self) -> None:
        self._approach_pos_history.clear()
        self._approach_last_seen_t = None
        # Seed the z setpoint from the reprojection estimate so the drone
        # immediately heads to approximately the right altitude, rather than
        # waiting for the edge-asymmetry servo to slowly accumulate a
        # correction from a near-zero signal at long range.
        gate = self.gates[self.current_gate]
        self._servo_z_setpoint = float(gate.position[2]) if gate is not None else None
        self._servo_z_settled = 0
        self._servo_xy_offset = None
        self._servo_xy_settled = 0

    def _approach_align_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Approach target = APPROACH_DIST in front of the gate, looking at it."""
        gate = self.gates[self.current_gate]
        assert gate is not None
        trav_dir = compute_traversal_direction(gate.position[:2], gate.yaw)
        align_xy = gate.position[:2] - config.APPROACH_DIST * trav_dir
        look_yaw = atan2(trav_dir[1], trav_dir[0])
        return (np.array([align_xy[0], align_xy[1], gate.position[2], look_yaw]),
                np.array([trav_dir[0], trav_dir[1], 0.0]))

    def _select_gate_detection(self, detections, drone_pos, roll, pitch, yaw):
        """Pick the detection closest to the projection of the current estimate."""
        gate = self.gates[self.current_gate]
        assert gate is not None
        Rc2w = R_cam2world(roll, pitch, yaw)
        cam_pos = cam_position_world(drone_pos, roll, pitch, yaw)
        p_cam = Rc2w.T @ (gate.position - cam_pos)
        if p_cam[2] <= 0.01:
            return max(detections, key=lambda d: d[1])[0]
        exp_u = config.FOCAL_PX * p_cam[0] / p_cam[2] + config.CAM_CX
        exp_v = config.FOCAL_PX * p_cam[1] / p_cam[2] + config.CAM_CY
        exp_center = np.array([exp_u, exp_v])

        best, best_dist = None, float("inf")
        for corners, _area, _idx in detections:
            d = float(np.linalg.norm(np.mean(corners, axis=0) - exp_center))
            if d < best_dist:
                best_dist, best = d, corners
        return best if best_dist <= 120.0 else None

    def _refine_with_reprojection(self, corners_2d: np.ndarray,
                                  drone_pos: np.ndarray,
                                  roll: float, pitch: float, yaw: float,
                                  ) -> bool:
        """Re-localize the gate from one new viewpoint and blend.

        Single-view reprojection from each new frame, warm-started with
        the current estimate. As the drone gets closer, the gate fills
        more pixels → corner localization improves → reprojection gets
        more accurate. No multi-view triangulation (which was unstable
        when the drone moves along the gate's normal axis).

        Bad updates are rejected against the YAW_SCAN prior (drift
        thresholds in config.REFINE_REJECT_*).
        """
        gate = self.gates[self.current_gate]
        assert gate is not None
        initial = [
            float(gate.position[0]), float(gate.position[1]),
            float(gate.position[2]), float(gate.yaw),
            float(gate.width),
        ]
        params, _cost = localize_gate_reprojection(
            corners_2d, drone_pos, roll, pitch, yaw,
            initial_guess=initial,
        )
        # Optimizer-output sanity (non-physical results)
        if params[2] < 0.3 or params[2] > 2.5:
            return False
        if params[4] < 0.15 or params[4] > 0.7:
            return False
        new_yaw = enforce_correct_yaw_orientation(
            params[0], params[1], params[3])
        # Same camera-vertical-bias compensation as in the yaw scan, applied
        # to the optimizer output (the warm-start prior is already corrected,
        # but the reprojection pulls z back toward the biased value).
        params[2] += config.GATE_Z_OFFSET

        new_pos = np.array(params[:3], dtype=np.float64)
        center_drift = float(np.linalg.norm(new_pos[:2] - gate.position[:2]))
        yaw_drift = abs(angle_diff(new_yaw, gate.yaw))
        width_drift = abs(params[4] - gate.width)
        if center_drift > config.REFINE_REJECT_CENTER_DRIFT:
            log.debug("Reprojection rejected (Δcenter=%.2fm)", center_drift)
            return False
        if yaw_drift > config.REFINE_REJECT_YAW_DRIFT:
            log.debug("Reprojection rejected (Δyaw=%.2frad)", yaw_drift)
            return False
        if width_drift > config.REFINE_REJECT_WIDTH_DRIFT:
            log.debug("Reprojection rejected (Δwidth=%.2fm)", width_drift)
            return False

        a = config.APPROACH_REFINE_ALPHA
        gate.position = (1 - a) * gate.position + a * new_pos
        gate.yaw = normalize_angle(
            gate.yaw + a * angle_diff(new_yaw, gate.yaw))
        gate.width = (1 - a) * gate.width + a * params[4]

        self._approach_pos_history.append(gate.position[:2].copy())
        if len(self._approach_pos_history) > config.CONVERGENCE_WINDOW:
            self._approach_pos_history.pop(0)
        return True

    def _approach_is_converged(self) -> bool:
        if len(self._approach_pos_history) < config.CONVERGENCE_MIN_SAMPLES:
            return False
        std_xy = np.array(self._approach_pos_history).std(axis=0)
        return float(std_xy.max()) < config.CONVERGENCE_STD_THRESHOLD

    def _visual_servo_offset(self, corners: np.ndarray,
                              drone_pos: np.ndarray,
                              roll: float, pitch: float, yaw: float,
                              gate_pos: np.ndarray) -> np.ndarray:
        """World-frame [dx, dy, dz] that centres the drone on the gate.

        Horizontal (y): edge-length asymmetry — scale-invariant, pitch has
        negligible effect on left/right symmetry.

        Vertical (z): pitch-compensated altitude error.
          Naively comparing v_c to CAM_CY assumes the optical axis is
          horizontal. As soon as the drone pitches forward during the
          approach (5–10° at NOMINAL_SPEED), the gate's image position
          shifts up by FOCAL_PX × tan(pitch) ≈ 15–30 px even though the
          drone is at the correct altitude, and the servo commands a
          phantom z correction of 0.1–0.2 m.

          Instead we project a synthetic point at (gate_x, gate_y, drone_z)
          — "where the gate would appear if the drone were at the correct
          altitude" — using the FULL camera-to-world rotation (so the same
          pitch that shifts v_c also shifts target_v by the same amount).
          The residual pixel_err is then purely an altitude offset, and
          multiplying by distance recovers the metric error:
            Δz = −K_V × (v_c − target_v) / FOCAL_PX × dist

        VISUAL_SERVO_K_V is a dimensionless feedback gain (~1.0).
        """
        tl, tr, br, bl = corners  # TL TR BR BL — image space, shape (2,) each

        # Horizontal: edge-length asymmetry (unchanged)
        l_left  = float(np.linalg.norm(bl - tl))
        l_right = float(np.linalg.norm(br - tr))
        avg_h = (l_left + l_right) * 0.5 + 1e-6
        e_h = (l_left - l_right) / avg_h   # dimensionless ∈ (−1, 1)
        delta_y = -config.VISUAL_SERVO_K_H * e_h

        # Vertical: pitch-compensated altitude error (see docstring above).
        Rc2w = R_cam2world(roll, pitch, yaw)
        cam_pos_w = cam_position_world(drone_pos, roll, pitch, yaw)
        p_cam = Rc2w.T @ (gate_pos - cam_pos_w)

        if p_cam[2] > 0.1:
            v_c = float(np.mean(corners[:, 1]))
            # Synthetic "gate at drone's current altitude" — its projection
            # is where v_c would land if the drone were already correctly
            # centred vertically, with the same pitch as right now.
            synthetic = np.array([gate_pos[0], gate_pos[1], drone_pos[2]])
            p_syn = Rc2w.T @ (synthetic - cam_pos_w)
            if p_syn[2] > 0.01:
                target_v = config.FOCAL_PX * p_syn[1] / p_syn[2] + config.CAM_CY
            else:
                target_v = config.CAM_CY  # synthetic point fell behind camera
            pixel_err = v_c - target_v
            dist = float(p_cam[2])
            delta_z = -config.VISUAL_SERVO_K_V * (pixel_err / config.FOCAL_PX) * dist
            # Camera-vertical-bias compensation. The servo is a closed loop
            # on a biased measurement: without this term it converges with
            # the drone ~GATE_Z_OFFSET below the gate centre. Adding a
            # constant shifts the loop's convergence point up by the same
            # amount (K_V=1 → delta_z is metric). See config.GATE_Z_OFFSET.
            delta_z += config.GATE_Z_OFFSET
            log.debug(
                "VisServo e_h=%.3f v_c=%.1f tgt_v=%.1f err=%.1fpx dist=%.2fm dz=%.3fm",
                e_h, v_c, target_v, pixel_err, dist, delta_z,
            )
        else:
            # Fallback: edge-length asymmetry (gate very close or behind)
            l_top    = float(np.linalg.norm(tr - tl))
            l_bottom = float(np.linalg.norm(br - bl))
            avg_v = (l_top + l_bottom) * 0.5 + 1e-6
            e_v = (l_top - l_bottom) / avg_v
            delta_z = -config.VISUAL_SERVO_K_V_FALLBACK * e_v
            log.debug("VisServo fallback e_h=%.3f e_v=%.3f dz=%.3fm", e_h, e_v, delta_z)

        # Clamp each axis
        clamp = config.VISUAL_SERVO_MAX_OFFSET
        delta_y = max(-clamp, min(clamp, delta_y))
        delta_z = max(-clamp, min(clamp, delta_z))

        # Rotate body lateral → world (ignore forward component)
        cy, sy = cos(yaw), sin(yaw)
        dx = -sy * delta_y
        dy =  cy * delta_y
        dz =  delta_z

        return np.array([dx, dy, dz])

    def _do_approach(self, sd: DroneState, image: np.ndarray | None,
                     drone_pos: np.ndarray) -> None:
        now = sd.t
        gate = self.gates[self.current_gate]
        assert gate is not None

        # 1. Detection + single-view reprojection refinement
        best_corners = None
        if image is not None:
            detections = detect_all_gate_contours(image)
            if detections:
                best_corners = self._select_gate_detection(
                    detections, drone_pos, sd.roll, sd.pitch, sd.yaw)

        if best_corners is not None and gate_pixel_width(best_corners) >= self.MIN_GATE_PX_WIDTH:
            self._approach_last_seen_t = now
            self._refine_with_reprojection(
                best_corners, drone_pos, sd.roll, sd.pitch, sd.yaw,
            )
        else:
            if (self._approach_last_seen_t is not None
                    and now - self._approach_last_seen_t > config.LOST_GATE_TIMEOUT_S):
                log.warning("Lost gate %d for >%.1fs — falling back",
                            self.current_gate, config.LOST_GATE_TIMEOUT_S)
                self._init_fallback_scan(self.current_gate)
                self._set_state(self.FALLBACK_SCAN)
                return

        # 2. Compute moving target = align-in-front-of-gate, look at gate.
        # xy comes from reprojection (accurate). z comes from the drone's
        # current z plus the visual servo's vertical correction — we never
        # trust the reprojected gate.z because it's biased by camera tilt.
        align_pose, _ = self._approach_align_pose()
        look_yaw = atan2(gate.position[1] - drone_pos[1],
                         gate.position[0] - drone_pos[0])

        # 3. Visual-servo correction: nudge target so left=right and top=bottom
        servo = np.zeros(3)
        if best_corners is not None and gate_pixel_width(best_corners) >= self.MIN_GATE_PX_WIDTH:
            servo = self._visual_servo_offset(best_corners, drone_pos,
                                              sd.roll, sd.pitch, sd.yaw,
                                              gate.position)
            # Accumulate z correction into a persistent setpoint so it doesn't
            # re-anchor to drone_pos[2] each tick (which prevents convergence).
            desired_z = drone_pos[2] + servo[2]
            if self._servo_z_setpoint is None:
                self._servo_z_setpoint = desired_z
            else:
                self._servo_z_setpoint = (0.55 * self._servo_z_setpoint
                                          + 0.45 * desired_z)
            self._servo_z_setpoint = float(np.clip(
                self._servo_z_setpoint, config.Z_BOUNDS[0], config.Z_BOUNDS[1]))
            # Accumulate the lateral correction into a persistent offset, the
            # same way as z, so the traverse target can reuse it.
            if self._servo_xy_offset is None:
                self._servo_xy_offset = servo[:2].copy()
            else:
                self._servo_xy_offset = (0.55 * self._servo_xy_offset
                                         + 0.45 * servo[:2])

        # 4. Issue go_to with servo-corrected target (rate-limited internally).
        # xy from reprojection; z from the persistent servo setpoint (immune to
        # reprojection's z bias and to re-anchoring on every tick).
        servo_xy = (self._servo_xy_offset if self._servo_xy_offset is not None
                    else np.zeros(2))
        target_x = align_pose[0] + servo_xy[0]
        target_y = align_pose[1] + servo_xy[1]
        target_z = (self._servo_z_setpoint if self._servo_z_setpoint is not None
                    else drone_pos[2])
        self._issue_goto(target_x, target_y, target_z,
                         look_yaw, cur_pos=drone_pos, cur_yaw=sd.yaw)

        # 5. Transition when arrived OR converged
        at_align = self._at_target(sd,
                                   target_xy=align_pose[:2],
                                   target_z=target_z,
                                   pos_tol=0.15)
        if at_align or self._approach_is_converged():
            log.info("Gate %d converged at %s yaw=%.2f -> TRAVERSE",
                     self.current_gate,
                     np.round(gate.position, 2), gate.yaw)
            self._traverse_started = False
            self._cmd_target = None
            self._set_state(self.ALIGN_AND_TRAVERSE)

    # ALIGN AND TRAVERSE
    def _do_align_traverse(self, sd: DroneState, image: np.ndarray | None,
                            drone_pos: np.ndarray) -> None:
        gate = self.gates[self.current_gate]
        assert gate is not None
        trav_dir = compute_traversal_direction(gate.position[:2], gate.yaw)
        trav_dir_3d = np.array([trav_dir[0], trav_dir[1], 0.0])

        # Thresholds for the settle checks before committing the traverse.
        _Z_SETTLE_THRESH = 0.06   # m — z servo correction small enough = centred
        _Z_SETTLE_FRAMES = 3      # consecutive frames required
        _XY_SETTLE_THRESH = 0.08  # m — lateral servo correction small enough
        _XY_SETTLE_FRAMES = 3

        if not self._traverse_started:
            align_pos_xy = gate.position[:2] - config.APPROACH_DIST * trav_dir
            look_yaw = atan2(trav_dir[1], trav_dir[0])

            # Visual-servo correction during the final alignment. The
            # reprojection-based xy may have a residual bias (gate.yaw is
            # imperfect → align direction off); the servo trims that.
            # For z we accumulate into the persistent setpoint (carried from
            # APPROACH) rather than re-anchoring to drone_pos[2] each tick.
            servo = np.zeros(3)
            if image is not None:
                detections = detect_all_gate_contours(image)
                if detections:
                    best = self._select_gate_detection(
                        detections, drone_pos, sd.roll, sd.pitch, sd.yaw)
                    if (best is not None
                            and gate_pixel_width(best) >= self.MIN_GATE_PX_WIDTH):
                        servo = self._visual_servo_offset(best, drone_pos,
                                                          sd.roll, sd.pitch, sd.yaw,
                                                          gate.position)
                        desired_z = drone_pos[2] + servo[2]
                        if self._servo_z_setpoint is None:
                            self._servo_z_setpoint = desired_z
                        else:
                            self._servo_z_setpoint = (0.55 * self._servo_z_setpoint
                                                      + 0.45 * desired_z)
                        self._servo_z_setpoint = float(np.clip(
                            self._servo_z_setpoint,
                            config.Z_BOUNDS[0], config.Z_BOUNDS[1]))
                        if abs(servo[2]) < _Z_SETTLE_THRESH:
                            self._servo_z_settled += 1
                        else:
                            self._servo_z_settled = 0
                        # Lateral correction — accumulate + settle, like z.
                        if self._servo_xy_offset is None:
                            self._servo_xy_offset = servo[:2].copy()
                        else:
                            self._servo_xy_offset = (0.55 * self._servo_xy_offset
                                                     + 0.45 * servo[:2])
                        if float(np.linalg.norm(servo[:2])) < _XY_SETTLE_THRESH:
                            self._servo_xy_settled += 1
                        else:
                            self._servo_xy_settled = 0

            servo_xy = (self._servo_xy_offset if self._servo_xy_offset is not None
                        else np.zeros(2))
            target_x = align_pos_xy[0] + servo_xy[0]
            target_y = align_pos_xy[1] + servo_xy[1]
            target_z = (self._servo_z_setpoint if self._servo_z_setpoint is not None
                        else drone_pos[2])
            self._issue_goto(target_x, target_y, target_z,
                             look_yaw, cur_pos=drone_pos, cur_yaw=sd.yaw)

            # Commit only after BOTH the z and the lateral servo have settled
            # (small correction = drone is centred on the gate axis).
            z_settled = self._servo_z_settled >= _Z_SETTLE_FRAMES
            xy_settled = self._servo_xy_settled >= _XY_SETTLE_FRAMES
            if (self._at_target(sd, target_xy=np.array([target_x, target_y]),
                                target_z=target_z,
                                target_yaw=look_yaw, pos_tol=0.10)
                    and z_settled and xy_settled):
                commit_z = (self._servo_z_setpoint if self._servo_z_setpoint is not None
                            else drone_pos[2])
                # Commanded target = TRAVERSE_OVERSHOOT past the gate plane,
                # in the traversal direction. Anchored on gate.position (not
                # drone_pos) so the drone actually flies through the gate.
                # The SAME lateral servo offset used for the align point is
                # applied here so the traverse flies straight along the
                # corrected gate axis instead of a diagonal through the
                # un-corrected reprojected centre (which clipped a leg).
                self._traverse_target = np.array([
                    gate.position[0] + config.TRAVERSE_OVERSHOOT * trav_dir[0]
                    + servo_xy[0],
                    gate.position[1] + config.TRAVERSE_OVERSHOOT * trav_dir[1]
                    + servo_xy[1],
                    commit_z,
                ])
                self._traverse_started = True
                self._cmd_target = None  # force re-issue
            return

        # Traversing
        target = self._traverse_target
        assert target is not None
        look_yaw = atan2(trav_dir[1], trav_dir[0])
        # Use a short duration so we actually fly fast through the plane.
        self._issue_goto(target[0], target[1], target[2], look_yaw,
                         duration=config.HLC_TRAVERSE_DURATION,
                         cur_pos=drone_pos, cur_yaw=sd.yaw)

        # Measure progress against the servo-corrected gate centre (the same
        # one the traverse target was built from), not the raw reprojection.
        center_xy = gate.position[:2]
        if self._servo_xy_offset is not None:
            center_xy = center_xy + self._servo_xy_offset
        forward = float(np.dot(drone_pos[:2] - center_xy, trav_dir))
        lateral = float(np.linalg.norm(
            (drone_pos[:2] - center_xy) - forward * trav_dir))
        # z_err vs the locked traverse z (set to the drone's z at the
        # moment servo had converged) — NOT vs gate.position[2] which
        # carries the reprojection bias.
        z_err = abs(drone_pos[2] - self._traverse_target[2])
        # Crossed once the drone is TRAVERSE_CLEAR_DIST past the gate plane,
        # so it fully clears the frame instead of stopping in the opening.
        # The HLC trajectory aims further out (TRAVERSE_OVERSHOOT) so the
        # drone still has non-zero velocity at the clear point.
        crossed = (forward >= config.TRAVERSE_CLEAR_DIST
                   and lateral < 0.3 and z_err < 0.3)
        if crossed:
            log.info("Gate %d crossed (fwd=%.2fm lat=%.2fm)",
                     self.current_gate, forward, lateral)
            self.current_gate += 1
            self._traverse_started = False
            self._cmd_target = None
            if self.current_gate >= config.NUM_GATES:
                log.info("All %d gates done — initiating LAND", config.NUM_GATES)
                self._init_land(drone_pos, sd.yaw)
                self._set_state(self.LAND)
            else:
                self._init_yaw_scan(self.current_gate, drone_pos, sd.yaw)
                self._set_state(self.YAW_SCAN)

    # FALLBACK SCAN
    def _init_fallback_scan(self, gate_idx: int) -> None:
        """Set up 3 stationary viewpoints, ordered OUTER → MID → INNER.

        Viewpoints sit on the radial at ``ang_hi`` of the PREVIOUS sector
        (= ``angular_bounds_for_segment(gate_idx)[1]``), NOT on
        ``ang_lo`` of the gate's own sector. The drone is then further
        from the gate by the full inter-sector gap, so a gate near the
        CW edge of its sector still fits inside the camera FOV during
        the yaw scan — at ``ang_lo`` of the gate's own sector the drone
        sits right next to such a gate and it overflows the frame.

        Each viewpoint runs a full stationary yaw_scan (see
        ``_yaw_scan_step``). If all 3 fail the gate is skipped (see
        ``_do_fallback_scan``).
        """
        boundary_angle = angular_bounds_for_segment(gate_idx)[1]
        radii = (
            config.OUTER_RADIUS,
            0.5 * (config.OUTER_RADIUS + config.INNER_RADIUS),
            config.INNER_RADIUS,
        )
        self._fallback_positions = [
            np.array([
                config.CIRCLE_CENTRE[0] + r * cos(boundary_angle),
                config.CIRCLE_CENTRE[1] + r * sin(boundary_angle),
            ])
            for r in radii
        ]
        self._fallback_pos_idx = 0
        self._fallback_substate = "MOVING"
        self._fallback_look_yaw = None
        self._cmd_target = None

    def _do_fallback_scan(self, sd: DroneState, image: np.ndarray | None,
                          drone_pos: np.ndarray) -> None:
        # Sub-FSM: MOVING (fly to the viewpoint) → SCANNING (stationary
        # yaw sweep) → either commit the detection or step to the next
        # viewpoint, with skip-the-gate as the final fallback.
        if self._fallback_substate == "MOVING":
            tgt = self._fallback_positions[self._fallback_pos_idx]
            # Lock the look yaw on entry. The old fallback recomputed it
            # from drone_pos every tick, which made it swing wildly as the
            # drone passed close to the look target. Computing it once
            # from the TARGET position pins it for the whole move.
            if self._fallback_look_yaw is None:
                gate_center = gate_search_center(self.current_gate)
                self._fallback_look_yaw = atan2(
                    gate_center[1] - tgt[1], gate_center[0] - tgt[0])
            self._issue_goto(tgt[0], tgt[1], config.EXPLORE_ALT,
                             self._fallback_look_yaw,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_xy=tgt,
                               target_z=config.EXPLORE_ALT,
                               target_yaw=self._fallback_look_yaw,
                               pos_tol=0.15):
                self._init_yaw_scan(self.current_gate, drone_pos, sd.yaw)
                self._fallback_substate = "SCANNING"
            return

        # SCANNING
        result = self._yaw_scan_step(sd, image, drone_pos)
        if result == "running":
            return
        if result == "found":
            p = self._scan_best_detection
            self.gates[self.current_gate] = GateEstimate(
                position=np.array([p[0], p[1], p[2]]),
                yaw=p[3], width=p[4],
            )
            log.info("Gate %d recovered during fallback at viewpoint %d (%s)",
                     self.current_gate, self._fallback_pos_idx,
                     np.round(self.gates[self.current_gate].position, 2))
            self._init_approach()
            self._set_state(self.APPROACH)
            return

        # No detection at this viewpoint — step to the next, or skip the gate.
        self._fallback_pos_idx += 1
        if self._fallback_pos_idx >= len(self._fallback_positions):
            log.warning("Fallback failed for gate %d at all %d viewpoints — "
                        "skipping gate", self.current_gate,
                        len(self._fallback_positions))
            self.current_gate += 1
            if self.current_gate >= config.NUM_GATES:
                log.info("All %d gates processed — initiating LAND",
                         config.NUM_GATES)
                self._init_land(drone_pos, sd.yaw)
                self._set_state(self.LAND)
            else:
                self._init_yaw_scan(self.current_gate, drone_pos, sd.yaw)
                self._set_state(self.YAW_SCAN)
            return

        # Next viewpoint: re-enter MOVING with a fresh look_yaw.
        self._fallback_substate = "MOVING"
        self._fallback_look_yaw = None
        self._cmd_target = None

    # LAND
    def _init_land(self, drone_pos: np.ndarray, drone_yaw: float) -> None:
        self._land_phase = 0
        self._land_issued = False
        self._cmd_target = None

    def _do_land(self, sd: DroneState, drone_pos: np.ndarray) -> None:
        if self._land_phase == 0:
            # Fly back to the take-off xy at current altitude.
            ret = self._land_return_xy
            look_yaw = atan2(ret[1] - drone_pos[1], ret[0] - drone_pos[0])
            # If already close, skip the look-yaw to avoid jittery commands.
            if np.linalg.norm(drone_pos[:2] - ret) < 0.2:
                look_yaw = sd.yaw
            self._issue_goto(ret[0], ret[1], config.EXPLORE_ALT, look_yaw,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_xy=ret, target_z=config.EXPLORE_ALT,
                               pos_tol=0.15):
                self._land_phase = 1
                self._cmd_target = None
        elif self._land_phase == 1:
            if not self._land_issued:
                self.cf.land(0.05, config.HLC_LAND_DURATION)
                self._land_issued = True
                self._cmd_t = time.monotonic()
            elif time.monotonic() - self._cmd_t >= config.HLC_LAND_DURATION + 0.5:
                self._set_state(self.DONE)
