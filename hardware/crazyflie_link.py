"""Crazyflie connection wrapper using cflib.

Owns:
- The radio link (Crazyradio PA → Crazyflie URI)
- A continuous Lighthouse pose log at 100 Hz
- A high-level commander wrapper (takeoff, go_to, land)
- Low-level position-setpoint streaming for trajectory racing

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
        try:
            self._scf.__enter__()
            self._cf = self._scf.cf
            self._configure_estimator()
            self._start_logging()
        except Exception:
            # Connection died mid-setup. Tear down so we don't leave the
            # radio link or background threads hanging — otherwise the
            # process never exits.
            log.error("Setup failed — closing radio link")
            try:
                self._scf.__exit__(None, None, None)
            except Exception:
                pass
            self._scf = None
            self._cf = None
            raise
        log.info("Crazyflie connected and logging started")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Emergency-land safety: if we're tearing down while the drone
        # is airborne (Ctrl+C, exception, or just script end), ALWAYS
        # try to land before closing the link. Without this, the radio
        # link drops, the firmware watchdog cuts motors, and the drone
        # falls. Skipped if z is near 0 (no point) or if cf isn't set.
        if self._cf is not None:
            try:
                s = self.get_state()
                airborne = s.z_global > 0.10
            except Exception:
                airborne = False

            if airborne:
                log.warning("Emergency land (z=%.2f m) before disconnect",
                            s.z_global)
                try:
                    self._cf.high_level_commander.land(
                        0.05, config.HLC_LAND_DURATION)
                    # Bounded wait — we MUST close the link eventually
                    # even if the land command somehow doesn't complete.
                    time.sleep(config.HLC_LAND_DURATION + 0.8)
                except Exception as e:
                    log.error("Emergency land failed: %s", e)

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
        has_flow, has_lh = self._detect_decks()
        self._cf.param.set_value("stabilizer.estimator", "2")  # Kalman

        # Flow Deck-only mode is prone to optical-flow oscillation when
        # the drone is stationary (Kalman over-trusts the noisy flow
        # signal, resulting in phantom drift). Tighten the noise model
        # to match the standard Bitcraze pattern. These params are
        # IRRELEVANT under Lighthouse and would confuse it — guard.
        if has_flow and not has_lh:
            try:
                self._cf.param.set_value("motion.adaptive", "0")
                self._cf.param.set_value("motion.flowStdFixed", "10.0")
                log.info("Flow Deck noise model tightened "
                         "(motion.adaptive=0, motion.flowStdFixed=10.0)")
            except Exception as e:
                log.warning("Could not set Flow Deck params: %s", e)

        # Reset estimator so the position snaps to the current take-off pose.
        self._cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self._cf.param.set_value("kalman.resetEstimation", "0")
        # High-level commander needs to be enabled for go_to/takeoff/land.
        self._cf.param.set_value("commander.enHighLevel", "1")

        # Convergence strategy depends on which deck provides position:
        # - Lighthouse gives absolute XY fixes → varXY converges to <0.001
        # - Flow Deck gives only relative motion → varXY stays high while
        #   the drone is stationary on the ground (no flow signal).
        # If we waited for varXY to settle on Flow Deck, we'd time out.
        if has_lh:
            self._wait_for_estimator_convergence(timeout=10.0)
        elif has_flow:
            # Standard Bitcraze pattern for Flow Deck only — give the
            # firmware ~2s to settle Z and ingest a few IMU samples.
            log.info("Flow Deck only — settling Kalman with a 2s sleep "
                     "(varXY won't converge without absolute fix)")
            time.sleep(2.0)
        else:
            raise RuntimeError(
                "No positioning deck detected (deck.bcFlow2=0 and "
                "deck.bcLighthouse4=0). Mount a Flow Deck or Lighthouse "
                "before flying — Kalman will not converge otherwise."
            )

    def _detect_decks(self) -> tuple[bool, bool]:
        """Return (has_flow, has_lighthouse) and log a one-line summary.

        With Flow Deck v2 the drone tracks position **relative** to its
        take-off point (0, 0, 0). With Lighthouse v4 it tracks position
        in the **absolute** room frame. The Kalman estimator works the
        same way with either — only the meaning of the coordinates and
        the convergence dynamics change.
        """
        assert self._cf is not None
        has_flow = False
        has_lh = False
        try:
            has_flow = int(self._cf.param.get_value("deck.bcFlow2")) != 0
        except (KeyError, AttributeError):
            pass
        try:
            has_lh = int(self._cf.param.get_value("deck.bcLighthouse4")) != 0
        except (KeyError, AttributeError):
            pass

        if has_lh and has_flow:
            log.info("Decks detected: Flow Deck v2 + Lighthouse v4 "
                     "(absolute positioning, room frame)")
        elif has_lh:
            log.info("Decks detected: Lighthouse v4 "
                     "(absolute positioning, room frame)")
        elif has_flow:
            log.info("Decks detected: Flow Deck v2 "
                     "(RELATIVE positioning — drone starts at (0,0,0))")
        else:
            log.warning("No positioning deck detected — Kalman will not "
                        "converge, takeoff will fail. Mount Flow Deck or "
                        "Lighthouse before flying.")
        return has_flow, has_lh

    def _wait_for_estimator_convergence(self, timeout: float = 10.0,
                                        threshold: float = 0.001) -> None:
        """Wait until the Kalman position variance has stabilized.

        Standard Bitcraze pattern: subscribe to ``kalman.varP{X,Y,Z}``,
        keep a 10-sample moving window per axis, and break once
        ``max - min < threshold`` on all three. Raises after ``timeout``
        seconds if the estimator never settles — which usually means
        Lighthouse is occluded or hasn't been started yet.
        """
        from cflib.crazyflie.log import LogConfig
        from cflib.crazyflie.syncLogger import SyncLogger

        cfg = LogConfig(name="Kalman Variance", period_in_ms=100)
        cfg.add_variable("kalman.varPX", "float")
        cfg.add_variable("kalman.varPY", "float")
        cfg.add_variable("kalman.varPZ", "float")

        var_x = [1.0] * 10
        var_y = [1.0] * 10
        var_z = [1.0] * 10

        log.info("Waiting for Kalman estimator to converge (timeout %.0fs)...",
                 timeout)
        deadline = time.monotonic() + timeout
        with SyncLogger(self._scf, cfg) as logger:
            for entry in logger:
                data = entry[1]
                var_x = var_x[1:] + [data["kalman.varPX"]]
                var_y = var_y[1:] + [data["kalman.varPY"]]
                var_z = var_z[1:] + [data["kalman.varPZ"]]

                if (max(var_x) - min(var_x) < threshold and
                        max(var_y) - min(var_y) < threshold and
                        max(var_z) - min(var_z) < threshold):
                    log.info("Kalman converged "
                             "(varX=%.4f varY=%.4f varZ=%.4f)",
                             var_x[-1], var_y[-1], var_z[-1])
                    return

                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Kalman did not converge within {timeout}s "
                        f"(latest var X/Y/Z = "
                        f"{var_x[-1]:.4f}/{var_y[-1]:.4f}/{var_z[-1]:.4f}). "
                        f"Check Lighthouse coverage at the take-off pose."
                    )

    def _start_logging(self) -> None:
        """Subscribe to Lighthouse pose + velocity at 100 Hz.

        cflib limits each log packet to 26 bytes, so 9 floats won't fit
        in one block. We split into two configs (pose: 6 floats × 4B = 24B,
        velocity: 3 floats × 4B = 12B), each with its own callback that
        updates a slice of the shared DroneState under the lock.
        """
        from cflib.crazyflie.log import LogConfig
        assert self._cf is not None

        pose_cfg = LogConfig(name="Pose", period_in_ms=10)
        for var in (
            "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw",
        ):
            pose_cfg.add_variable(var, "float")
        self._cf.log.add_config(pose_cfg)
        pose_cfg.data_received_cb.add_callback(self._on_pose_log)
        pose_cfg.start()

        vel_cfg = LogConfig(name="Vel", period_in_ms=10)
        for var in (
            "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
        ):
            vel_cfg.add_variable(var, "float")
        self._cf.log.add_config(vel_cfg)
        vel_cfg.data_received_cb.add_callback(self._on_vel_log)
        vel_cfg.start()

    def _stamp(self) -> float:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    def _on_pose_log(self, _ts: int, data: dict[str, Any], _logconf: Any) -> None:
        t = self._stamp()
        with self._state_lock:
            self._state.t = t
            self._state.x_global = data["stateEstimate.x"]
            self._state.y_global = data["stateEstimate.y"]
            self._state.z_global = data["stateEstimate.z"]
            self._state.roll = math.radians(data["stateEstimate.roll"])
            self._state.pitch = math.radians(data["stateEstimate.pitch"])
            self._state.yaw = math.radians(data["stateEstimate.yaw"])

    def _on_vel_log(self, _ts: int, data: dict[str, Any], _logconf: Any) -> None:
        with self._state_lock:
            self._state.v_x = data["stateEstimate.vx"]
            self._state.v_y = data["stateEstimate.vy"]
            self._state.v_z = data["stateEstimate.vz"]

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

    # Low-level setpoint streaming (used for trajectory racing)
    def send_position_setpoint(
        self, x: float, y: float, z: float, yaw: float,
    ) -> None:
        """Stream one absolute position setpoint (low-level commander).

        Bypasses the High-Level Commander: the firmware position
        controller tracks this exact setpoint, so the drone follows our
        planned trajectory point-by-point instead of letting the HLC
        rebuild its own min-snap path (which cuts corners).

        Must be streamed continuously (>2 Hz) — if the firmware setpoint
        watchdog goes ~500 ms without a packet it cuts the motors.

        ``yaw`` is in radians (converted to the degrees cflib expects).
        """
        assert self._cf is not None
        self._cf.commander.send_position_setpoint(x, y, z, math.degrees(yaw))

    def notify_setpoint_stop(self, remain_valid_ms: int = 200) -> None:
        """Hand control back to the High-Level Commander after streaming.

        Call once after the last low-level setpoint and before re-issuing
        any HLC command (``go_to``/``land``); otherwise the HLC stays
        suppressed by the stale setpoint. ``remain_valid_ms`` keeps the
        last streamed setpoint active for that long, bridging the gap
        until the next HLC command takes over.
        """
        assert self._cf is not None
        self._cf.commander.send_notify_setpoint_stop(remain_valid_ms)
