"""Racing run entry point (laps 2 & 3).

Connects to the Crazyflie, runs the RaceFSM at 20 Hz and logs the flight
trace to a timestamped CSV. The racing FSM flies a pre-planned min-jerk
trajectory through known gates and never uses the camera, so the AI Deck
is not opened here at all — the run works whether or not the AI Deck is
mounted on the drone.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hardware.crazyflie_link import CrazyflieLink
from race_fsm import RaceFSM

log = logging.getLogger(__name__)

TICK_PERIOD: float = 0.05  # 20 Hz — plenty for the setpoint stream
MISSION_TIMEOUT_S: float = 240.0  # safety: assignment hard cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=str(
        Path(__file__).resolve().parent / "logs"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "trace.csv"
    log.info("Run logs → %s", run_dir)

    with CrazyflieLink() as cf:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_mono", "t_drone", "state", "gate",
                "x", "y", "z", "roll", "pitch", "yaw",
                "vx", "vy", "vz",
            ])

            fsm = RaceFSM(cf)
            t_start = time.monotonic()
            next_t = t_start

            try:
                while True:
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t += TICK_PERIOD

                    sd = cf.get_state()
                    still_running = fsm.tick(sd)

                    writer.writerow([
                        f"{now - t_start:.3f}", f"{sd.t:.3f}",
                        fsm.state, fsm.current_gate,
                        f"{sd.x_global:.3f}", f"{sd.y_global:.3f}",
                        f"{sd.z_global:.3f}",
                        f"{sd.roll:.3f}", f"{sd.pitch:.3f}", f"{sd.yaw:.3f}",
                        f"{sd.v_x:.3f}", f"{sd.v_y:.3f}", f"{sd.v_z:.3f}",
                    ])

                    if not still_running:
                        log.info("Mission complete")
                        break
                    if time.monotonic() - t_start > MISSION_TIMEOUT_S:
                        log.error("Mission timeout (%.0fs) — aborting",
                                  MISSION_TIMEOUT_S)
                        break
            except KeyboardInterrupt:
                log.warning("Interrupted by user")
            # Whatever the exit path (normal, timeout, Ctrl+C, exception),
            # CrazyflieLink.__exit__ runs the emergency-land procedure
            # before closing the radio link.


if __name__ == "__main__":
    main()
