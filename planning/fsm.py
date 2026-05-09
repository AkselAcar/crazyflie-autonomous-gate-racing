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
    REISSUE_POS_THRESHOLD = 0.05   # m
    REISSUE_YAW_THRESHOLD = 0.05   # rad
    MIN_REISSUE_INTERVAL = 0.4     # s

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

        # Traverse state
        self._traverse_target: np.ndarray | None = None
        self._traverse_started: bool = False

        # Fallback scan state
        self._fallback_inner_pos: np.ndarray | None = None
        self._fallback_outer_pos: np.ndarray | None = None
        self._fallback_look_target: np.ndarray | None = None
        self._fallback_phase: int = 0

        # Landing state
        self._land_phase: int = 0
        self._land_return_xy: np.ndarray = config.TAKE_OFF_POS[:2]

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
            self._do_align_traverse(sd, drone_pos)
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
        return speed < self.ARRIVAL_VEL_TOL * 3  # generous

    # TAKEOFF
    def _do_takeoff(self, sd: DroneState, drone_pos: np.ndarray) -> None:
        if not self._takeoff_issued:
            log.info("Resting pose: x=%.2f y=%.2f z=%.2f yaw=%.2f",
                     sd.x_global, sd.y_global, sd.z_global, sd.yaw)
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

    def _do_yaw_scan(self, sd: DroneState, image: np.ndarray | None,
                     drone_pos: np.ndarray) -> None:
        # Run the detector regardless of phase.
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
        elif self._scan_phase == 1:
            duration = self._scan_duration(sd.yaw, self._scan_second_target)
            self._issue_goto(anchor[0], anchor[1], z, self._scan_second_target,
                             duration=duration,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_yaw=self._scan_second_target):
                self._scan_phase = 2
        else:
            # Scan complete — pick best detection or fallback.
            if self._scan_best_detection is not None:
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
            if dist_to_region < -1.0:
                continue
            params[3] = enforce_correct_yaw_orientation(
                params[0], params[1], params[3])
            if cost < self._scan_best_cost:
                self._scan_best_cost = cost
                self._scan_best_detection = params

    # APPROACH (multi-view triangulation)
    def _init_approach(self) -> None:
        self._approach_pos_history.clear()
        self._approach_last_seen_t = None

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

        # 2. Compute moving target = align-in-front-of-gate, look at gate
        align_pose, _ = self._approach_align_pose()
        look_yaw = atan2(gate.position[1] - drone_pos[1],
                         gate.position[0] - drone_pos[0])

        # 3. Issue go_to (rate-limited internally)
        self._issue_goto(align_pose[0], align_pose[1], align_pose[2],
                         look_yaw, cur_pos=drone_pos, cur_yaw=sd.yaw)

        # 4. Transition when arrived OR converged
        at_align = self._at_target(sd,
                                   target_xy=align_pose[:2],
                                   target_z=align_pose[2],
                                   pos_tol=0.15)
        if at_align or self._approach_is_converged():
            log.info("Gate %d converged at %s yaw=%.2f -> TRAVERSE",
                     self.current_gate,
                     np.round(gate.position, 2), gate.yaw)
            self._traverse_started = False
            self._cmd_target = None
            self._set_state(self.ALIGN_AND_TRAVERSE)

    # ALIGN AND TRAVERSE
    def _do_align_traverse(self, sd: DroneState, drone_pos: np.ndarray) -> None:
        gate = self.gates[self.current_gate]
        assert gate is not None
        trav_dir = compute_traversal_direction(gate.position[:2], gate.yaw)
        trav_dir_3d = np.array([trav_dir[0], trav_dir[1], 0.0])

        if not self._traverse_started:
            align_pos = gate.position - config.APPROACH_DIST * trav_dir_3d
            align_pos[2] = gate.position[2]
            look_yaw = atan2(trav_dir[1], trav_dir[0])

            self._issue_goto(align_pos[0], align_pos[1], align_pos[2],
                             look_yaw, cur_pos=drone_pos, cur_yaw=sd.yaw)

            if self._at_target(sd, target_xy=align_pos[:2],
                               target_z=align_pos[2],
                               target_yaw=look_yaw, pos_tol=0.10):
                self._traverse_target = (
                    gate.position + config.TRAVERSE_COMMAND_OVERSHOOT * trav_dir_3d)
                self._traverse_target[2] = gate.position[2]
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

        forward = float(np.dot(drone_pos[:2] - gate.position[:2], trav_dir))
        lateral = float(np.linalg.norm(
            (drone_pos[:2] - gate.position[:2]) - forward * trav_dir))
        z_err = abs(drone_pos[2] - gate.position[2])
        crossed = (forward >= config.TRAVERSE_OVERSHOOT
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
        seg = gate_idx + 1
        boundary_angle = angular_bounds_for_segment(seg)[0]
        self._fallback_inner_pos = np.array([
            config.CIRCLE_CENTRE[0] + config.INNER_RADIUS * cos(boundary_angle),
            config.CIRCLE_CENTRE[1] + config.INNER_RADIUS * sin(boundary_angle),
        ])
        self._fallback_outer_pos = np.array([
            config.CIRCLE_CENTRE[0] + config.OUTER_RADIUS * cos(boundary_angle),
            config.CIRCLE_CENTRE[1] + config.OUTER_RADIUS * sin(boundary_angle),
        ])
        self._fallback_look_target = gate_search_center(gate_idx)
        self._fallback_phase = 0
        self._scan_best_detection = None
        self._scan_best_cost = float("inf")
        self._cmd_target = None

    def _do_fallback_scan(self, sd: DroneState, image: np.ndarray | None,
                          drone_pos: np.ndarray) -> None:
        if image is not None:
            self._try_detect_gate_during_scan(image, drone_pos,
                                              sd.roll, sd.pitch, sd.yaw)

        gate = self.gates[self.current_gate]
        target_z = gate.position[2] if gate is not None else config.EXPLORE_ALT

        look = self._fallback_look_target
        assert look is not None
        look_yaw = atan2(look[1] - drone_pos[1], look[0] - drone_pos[0])

        if self._fallback_phase == 0:
            tgt = self._fallback_inner_pos
            assert tgt is not None
            self._issue_goto(tgt[0], tgt[1], target_z, look_yaw,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_xy=tgt, target_z=target_z, pos_tol=0.15):
                self._fallback_phase = 1
                self._cmd_target = None
        elif self._fallback_phase == 1:
            tgt = self._fallback_outer_pos
            assert tgt is not None
            self._issue_goto(tgt[0], tgt[1], target_z, look_yaw,
                             cur_pos=drone_pos, cur_yaw=sd.yaw)
            if self._at_target(sd, target_xy=tgt, target_z=target_z, pos_tol=0.15):
                self._fallback_phase = 2
        else:
            if self._scan_best_detection is not None:
                p = self._scan_best_detection
                self.gates[self.current_gate] = GateEstimate(
                    position=np.array([p[0], p[1], p[2]]),
                    yaw=p[3], width=p[4],
                )
                log.info("Gate %d recovered during fallback at %s",
                         self.current_gate,
                         np.round(self.gates[self.current_gate].position, 2))
                self._init_approach()
                self._set_state(self.APPROACH)
            else:
                log.warning("Fallback failed for gate %d, retrying yaw scan",
                            self.current_gate)
                self._init_yaw_scan(self.current_gate, drone_pos, sd.yaw)
                self._set_state(self.YAW_SCAN)

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
