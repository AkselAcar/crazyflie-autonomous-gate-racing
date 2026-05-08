"""Smoke test: connect, take off, hover, land.

Run from C:\\Users\\niels\\Desktop\\Aerial_Robotics\\Hardware_Project:

    python debug/test_takeoff_land.py

This is the first hardware test to run after wiring everything up. It
exercises only the radio link and the High-Level Commander (no camera,
no FSM). Use it to verify Lighthouse convergence and basic flight
before moving on to image capture or autonomous flight.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running this script directly from Hardware_Project/ — make the
# project root (this file's grandparent) the first entry on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from hardware.crazyflie_link import CrazyflieLink

log = logging.getLogger(__name__)

HOVER_HEIGHT: float = 0.5    # m — low and safe
HOVER_TIME: float = 3.0      # s
# HLC takeoff with too gentle a climb rate (e.g. 0.5m / 2.0s = 0.25 m/s)
# leaves the drone "hovering on the ground": motors spin to follow the
# trajectory but the commanded thrust isn't enough to break free. Use a
# shorter duration so the climb rate is ~0.5 m/s, which actually lifts.
TAKEOFF_DURATION: float = 1.0  # s


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with CrazyflieLink() as cf:
        # Print the resting pose (Lighthouse absolute, or Flow Deck (0,0,0)).
        s = cf.get_state()
        log.info("Resting pose: x=%.2f y=%.2f z=%.2f yaw=%.2f rad",
                 s.x_global, s.y_global, s.z_global, s.yaw)

        cf.takeoff(HOVER_HEIGHT, TAKEOFF_DURATION)
        time.sleep(TAKEOFF_DURATION)

        # Log pose every 0.5s during hover so we can see Flow Deck drift live.
        n_samples = int(HOVER_TIME / 0.5)
        for _ in range(n_samples):
            time.sleep(0.5)
            s = cf.get_state()
            log.info("  hover: x=%+.2f y=%+.2f z=%.2f yaw=%+.2f",
                     s.x_global, s.y_global, s.z_global, s.yaw)

        cf.land(0.05, config.HLC_LAND_DURATION)
        time.sleep(config.HLC_LAND_DURATION + 0.5)

    log.info("Smoke test done")


if __name__ == "__main__":
    main()
