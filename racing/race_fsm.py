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
import sys
import time
from math import atan2
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from hardware.crazyflie_link import CrazyflieLink, DroneState
from planning.geometry import angle_diff
from motion_planner import MotionPlanner3D

log = logging.getLogger(__name__)


class RaceFSM:
    """Finite-state machine that flies lap 2&3 and lands."""

    # States
    TAKEOFF = 0
    TRAJECTORY_PLANNING = 1
    RACING = 2
    LAND = 3
    DONE = 4

    _STATE_NAMES = {
        TAKEOFF: "TAKEOFF", TRAJECTORY_PLANNING: "TRAJECTORY_PLANNING", RACING: "RACING",
        LAND: "LAND", DONE: "DONE",
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

    def __init__(self, cf: CrazyflieLink) -> None:
        self.cf = cf
        self.state: int = self.TAKEOFF
        self.current_gate: int = 0

        # HLC bookkeeping
        self._cmd_target: np.ndarray | None = None  # [x, y, z, yaw]
        self._cmd_t: float = 0.0
        self._takeoff_issued: bool = False
        self._land_issued: bool = False

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
        elif self.state == self.TRAJECTORY_PLANNING:
            self._do_trajectory_planning(sd, drone_pos)
        elif self.state == self.RACING:
            self._do_racing(sd)
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

    def _do_trajectory_planning(self, sd: DroneState,
                                drone_pos: np.ndarray) -> None:
        # Plan the racing trajectory from the current pose through the
        # known gate positions. Runs synchronously: the drone holds its
        # hover via the HLC while the polynomial solve completes.
        self.motion_planner = MotionPlanner3D(
            tuple(drone_pos), config.GATE_POS_CSV_PATH,
            verbose=config.MOTION_PLANNER_VERBOSE)
        self.lap_start_time = time.monotonic()
        self._set_state(self.RACING)

    def _do_racing(self, sd: DroneState) -> None:
        traj = self.motion_planner.trajectory_setpoints
        time_vector = self.motion_planner.time_setpoints

        # Setpoint index matching the elapsed trajectory time.
        elapsed_time = time.monotonic() - self.lap_start_time
        idx = int(np.searchsorted(time_vector, elapsed_time))

        if idx >= len(traj):
            # Trajectory complete → land.
            self._init_land()
            self._set_state(self.LAND)
            return

        # Local path speed, used to size a speed-dependent lookahead.
        sp_current = traj[idx]
        sp_next = traj[min(idx + 1, len(traj) - 1)]
        ds = float(np.linalg.norm(sp_next[:2] - sp_current[:2]))
        dt_planner = (time_vector[min(idx + 1, len(time_vector) - 1)]
                      - time_vector[idx])
        path_velocity = ds / dt_planner if dt_planner > 0 else 0.0

        # Lookahead distance in metres → number of trajectory setpoints.
        # higher multiplier = cuts corners tighter, lower = swings wide.
        lookahead_m = (config.BASE_LOOKAHEAD
                       + path_velocity * config.LOOKAHEAD_SPEED_MULTIPLIER)
        lookahead_idx = int(lookahead_m * config.PTS_PER_METER)
        # Ramp the lookahead in over the first second to avoid a big jump.
        if elapsed_time < 1.0:
            lookahead_idx = int(lookahead_idx * elapsed_time)

        pos_idx = min(idx + lookahead_idx, len(traj) - 1)
        sp = traj[pos_idx]
        yaw_idx = min(pos_idx + 5, len(traj) - 1)
        sp_future = traj[yaw_idx]

        # Yaw faces along the path; hold current yaw when near-stationary.
        sp_dir = sp_future - sp
        if np.linalg.norm(sp_dir[:2]) > 0.01:
            yaw_sp = atan2(sp_dir[1], sp_dir[0])
        else:
            yaw_sp = sd.yaw

        drone_pos = np.array([sd.x_global, sd.y_global, sd.z_global])
        self._issue_goto(float(sp[0]), float(sp[1]), float(sp[2]), yaw_sp,
                         cur_pos=drone_pos, cur_yaw=sd.yaw)

    
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
            self._set_state(self.TRAJECTORY_PLANNING)

    # LAND
    def _init_land(self) -> None:
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
