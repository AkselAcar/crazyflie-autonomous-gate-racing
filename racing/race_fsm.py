"""Racing FSM for laps 2 & 3: fly the planned trajectory, then land.

Pipeline:
    TAKEOFF → TRAJECTORY_PLANNING → RACING → LAND → DONE

- TAKEOFF and LAND use the High-Level Commander (takeoff/land): the
  firmware builds a min-snap path for those simple vertical moves. The
  racing trajectory ends above the take-off pad, so LAND just descends
  in place — no fly-back leg.
- RACING streams low-level full-state setpoints (position plus
  velocity/acceleration feedforward) via ``cf.send_full_state_setpoint``.
  The motion planner's min-jerk trajectory is sampled by elapsed time so
  the drone tracks the planned polynomial faithfully — no lookahead, no
  corner-cutting, and the feedforward removes the controller lag. On
  completion ``cf.notify_setpoint_stop`` hands control back to the HLC
  for landing.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from hardware.crazyflie_link import CrazyflieLink, DroneState
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

    def __init__(self, cf: CrazyflieLink) -> None:
        self.cf = cf
        # Mellinger controller — needed to exploit the velocity/accel
        # feedforward streamed during RACING (see _do_racing).
        self.cf.set_controller(config.RACE_CONTROLLER)
        # Override the Mellinger's vehicle mass to compensate the constant
        # z-bias observed at default settings. Volatile param — must be
        # set every session (see CrazyflieLink.set_mellinger_mass).
        self.cf.set_mellinger_mass(config.RACE_DRONE_MASS)
        self.state: int = self.TAKEOFF
        self.current_gate: int = 0

        # HLC bookkeeping. _cmd_t timestamps the takeoff and land commands.
        self._cmd_t: float = 0.0
        self._takeoff_issued: bool = False
        self._land_issued: bool = False

        # Actual flown trajectory recorded for the post-mission overlay
        # against the planned trajectory: one estimator pose plus its
        # time.monotonic() timestamp per tick. The timestamps let the
        # overlay resample the plan onto the same time base, so flown and
        # planned points can be compared instant-by-instant.
        self._flown_path: list[np.ndarray] = []
        self._flown_times: list[float] = []
        # Index range in _flown_path covering the racing trajectory. The
        # planned trajectory spans the race start pose to the last
        # setpoint, so the post-mission overlay slices [start:end] — the
        # takeoff/planning poses before and the landing poses after are
        # excluded to keep both curves coherent.
        self._race_path_start_idx: int = 0
        self._race_path_end_idx: int | None = None

    # Public API
    def tick(self, sd: DroneState) -> bool:
        """Drive the FSM one step. Returns False when mission complete."""
        if self.state == self.DONE:
            return False

        drone_pos = np.array([sd.x_global, sd.y_global, sd.z_global])
        self._flown_path.append(drone_pos)
        self._flown_times.append(time.monotonic())

        if self.state == self.TAKEOFF:
            self._do_takeoff(sd, drone_pos)
        elif self.state == self.TRAJECTORY_PLANNING:
            self._do_trajectory_planning(sd, drone_pos)
        elif self.state == self.RACING:
            self._do_racing(sd)
        elif self.state == self.LAND:
            self._do_land()

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
        # The planned trajectory's first waypoint is this exact pose, which
        # tick() already appended to _flown_path this tick — mark its index
        # so the overlay plots the flown path from the same starting point.
        self._race_path_start_idx = max(0, len(self._flown_path) - 1)
        self.lap_start_time = time.monotonic()
        self._set_state(self.RACING)

    def _do_racing(self, sd: DroneState) -> None:
        traj = self.motion_planner.trajectory_setpoints
        vel = self.motion_planner.trajectory_velocities
        acc = self.motion_planner.trajectory_accelerations
        time_vector = self.motion_planner.time_setpoints

        # Index the trajectory by elapsed time and stream the setpoint
        # for *this* instant — no lookahead. The drone tracks the planned
        # polynomial faithfully instead of chord-cutting corners.
        elapsed_time = time.monotonic() - self.lap_start_time
        idx = int(np.searchsorted(time_vector, elapsed_time))

        if idx >= len(traj):
            # Trajectory complete → hand control back to the HLC and land.
            self.cf.notify_setpoint_stop()
            self._race_path_end_idx = len(self._flown_path)
            self._init_land()
            self._set_state(self.LAND)
            return

        sp = traj[idx]  # [x, y, z, yaw] — yaw already faces along the path
        # Full-state setpoint: position plus velocity/acceleration
        # feedforward so the Mellinger controller anticipates the moving
        # setpoint instead of lagging it.
        self.cf.send_full_state_setpoint(
            sp[:3], vel[idx], acc[idx], float(sp[3]))


    # TAKEOFF
    def _do_takeoff(self, sd: DroneState, drone_pos: np.ndarray) -> None:
        if not self._takeoff_issued:
            log.info("Resting pose: x=%.2f y=%.2f z=%.2f yaw=%.2f",
                     sd.x_global, sd.y_global, sd.z_global, sd.yaw)
            self.cf.takeoff(config.EXPLORE_ALT, config.HLC_TAKEOFF_DURATION)
            self._takeoff_issued = True
            self._cmd_t = time.monotonic()
            return
        # The HLC takeoff is a fixed-duration maneuver — transition once it
        # has run (plus a short settle margin). We deliberately do NOT gate
        # on reaching EXPLORE_ALT exactly: the controller may leave a few
        # cm of steady-state error, and the racing trajectory is planned
        # from the actual post-takeoff pose anyway, so the precise altitude
        # does not matter.
        if time.monotonic() - self._cmd_t < config.HLC_TAKEOFF_DURATION + 0.5:
            return
        self._set_state(self.TRAJECTORY_PLANNING)

    # LAND
    def _init_land(self) -> None:
        self._land_issued = False

    def _do_land(self) -> None:
        # The racing trajectory ends above the take-off pad, so the drone
        # is already in place — just descend with the HLC, in place.
        if not self._land_issued:
            self.cf.land(0.05, config.HLC_LAND_DURATION)
            self._land_issued = True
            self._cmd_t = time.monotonic()
            return
        if time.monotonic() - self._cmd_t < config.HLC_LAND_DURATION + 0.5:
            return
        self._set_state(self.DONE)
        # Overlay only the racing portion of the flown path so it spans the
        # same range as the planned trajectory: from the race start pose to
        # the last setpoint (landing excluded). Timestamps are rebased to
        # trajectory time (0 at race start) to match the planner's time axis.
        i0 = self._race_path_start_idx
        i1 = (self._race_path_end_idx
              if self._race_path_end_idx is not None
              else len(self._flown_path))
        flown = np.array(self._flown_path[i0:i1])
        flown_times = (np.array(self._flown_times[i0:i1])
                       - self.lap_start_time)
        self.motion_planner.plot(flown, flown_times)
