"""Crazyflie connection wrapper using cflib.

Owns:
- The radio link (Crazyradio PA → Crazyflie URI)
- A continuous Lighthouse pose log at 100 Hz
- A high-level commander wrapper (takeoff, go_to, land)

The state dict returned by `get_state()` mirrors the keys used by the
Webots controller as closely as possible (`x_global`, `y_global`,
`z_global`, `roll`, `pitch`, `yaw`, `t`, `v_x`, ...) so the FSM port
stays minimal — only the source of the data changes.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import config

# cflib is only needed when actually connecting to the drone. Importing
# it lazily lets the rest of the package (FSM, geometry, perception) be
# imported and unit-tested in environments where cflib isn't installed
# (e.g. on a machine without Crazyradio drivers).

log = logging.getLogger(__name__)


@dataclass
class DroneState:
    """Latest Lighthouse + IMU snapshot."""

    t: float = 0.0
    x_global: float = 0.0
    y_global: float = 0.0
    z_global: float = 0.0
    roll: float = 0.0   # rad
    pitch: float = 0.0  # rad
    yaw: float = 0.0    # rad
    v_x: float = 0.0
    v_y: float = 0.0
    v_z: float = 0.0


class CrazyflieLink:
    """High-level interface around a connected Crazyflie.

    Use as a context manager:

        with CrazyflieLink() as cf:
            cf.takeoff()
            time.sleep(2)
            cf.go_to(0, 0, 1.0, 0.0, duration=2.0)
            ...
            cf.land()
    """

    def __init__(self, uri: str = config.CF_URI) -> None:
        self._uri = uri
        self._scf: Any | None = None  # SyncCrazyflie
        self._cf: Any | None = None   # Crazyflie
        self._state = DroneState()
        self._state_lock = threading.Lock()
        self._t0: float | None = None

    # Lifecycle
    def __enter__(self) -> "CrazyflieLink":
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

        cflib.crtp.init_drivers()
        log.info("Connecting to Crazyflie at %s", self._uri)
        self._scf = SyncCrazyflie(self._uri, cf=Crazyflie(rw_cache="./cache"))
        self._scf.__enter__()
        self._cf = self._scf.cf
        self._configure_estimator()
        self._start_logging()
        log.info("Crazyflie connected and logging started")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._scf is not None and self._cf is not None:
            try:
                self._cf.high_level_commander.stop()
            except Exception as e:
                log.warning("HLC stop failed during disconnect: %s", e)
            self._scf.__exit__(exc_type, exc, tb)
        log.info("Crazyflie disconnected")

    def _configure_estimator(self) -> None:
        """Activate the Kalman estimator and the High-Level Commander."""
        assert self._cf is not None
        self._cf.param.set_value("stabilizer.estimator", "2")  # Kalman
        # Reset estimator so the position snaps to the current take-off pose.
        self._cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self._cf.param.set_value("kalman.resetEstimation", "0")
        # High-level commander needs to be enabled for go_to/takeoff/land.
        self._cf.param.set_value("commander.enHighLevel", "1")
        # Wait for Lighthouse fix to converge before any motion.
        time.sleep(2.0)

    def _start_logging(self) -> None:
        from cflib.crazyflie.log import LogConfig
        assert self._cf is not None
        log_cfg = LogConfig(name="State", period_in_ms=10)
        for var in (
            "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw",
            "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
        ):
            log_cfg.add_variable(var, "float")
        self._cf.log.add_config(log_cfg)
        log_cfg.data_received_cb.add_callback(self._on_log)
        log_cfg.start()

    def _on_log(self, _ts: int, data: dict[str, Any], _logconf: Any) -> None:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        new = DroneState(
            t=now - self._t0,
            x_global=data["stateEstimate.x"],
            y_global=data["stateEstimate.y"],
            z_global=data["stateEstimate.z"],
            roll=math.radians(data["stateEstimate.roll"]),
            pitch=math.radians(data["stateEstimate.pitch"]),
            yaw=math.radians(data["stateEstimate.yaw"]),
            v_x=data["stateEstimate.vx"],
            v_y=data["stateEstimate.vy"],
            v_z=data["stateEstimate.vz"],
        )
        with self._state_lock:
            self._state = new

    # Public API
    def get_state(self) -> DroneState:
        """Snapshot of the latest pose + velocity (thread-safe)."""
        with self._state_lock:
            return DroneState(**self._state.__dict__)

    @property
    def hlc(self):
        assert self._cf is not None
        return self._cf.high_level_commander

    def takeoff(
        self,
        height: float = config.EXPLORE_ALT,
        duration: float = config.HLC_TAKEOFF_DURATION,
    ) -> None:
        log.info("HLC takeoff to %.2f m over %.1fs", height, duration)
        self.hlc.takeoff(height, duration)

    def go_to(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration: float,
        relative: bool = False,
    ) -> None:
        """Issue an absolute (or relative) min-snap go_to.

        The firmware generates a smooth polynomial trajectory between the
        current state and the target — no Python-side smoother needed.
        """
        self.hlc.go_to(x, y, z, yaw, duration, relative=relative)

    def land(
        self,
        height: float = 0.05,
        duration: float = config.HLC_LAND_DURATION,
    ) -> None:
        log.info("HLC land to %.2f m over %.1fs", height, duration)
        self.hlc.land(height, duration)
