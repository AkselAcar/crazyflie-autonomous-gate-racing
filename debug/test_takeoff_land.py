"""Smoke test: connect, take off, hover, land.

Run from C:\\Users\\niels\\Desktop\\Aerial_Robotics:

    python -m Hardware_Project.debug.test_takeoff_land

This is the first hardware test to run after wiring everything up. It
exercises only the radio link and the High-Level Commander (no camera,
no FSM). Use it to verify Lighthouse convergence and basic flight
before moving on to image capture or autonomous flight.
"""

from __future__ import annotations

import logging
import time

from Hardware_Project import config
from Hardware_Project.hardware.crazyflie_link import CrazyflieLink

log = logging.getLogger(__name__)

HOVER_HEIGHT: float = 0.5  # m — low and safe
HOVER_TIME: float = 3.0    # s


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with CrazyflieLink() as cf:
        # Print the resting Lighthouse pose to verify we got a fix.
        s = cf.get_state()
        log.info("Resting pose: x=%.2f y=%.2f z=%.2f yaw=%.2f rad",
                 s.x_global, s.y_global, s.z_global, s.yaw)

        cf.takeoff(HOVER_HEIGHT, config.HLC_TAKEOFF_DURATION)
        time.sleep(config.HLC_TAKEOFF_DURATION + HOVER_TIME)

        s = cf.get_state()
        log.info("Hover pose: x=%.2f y=%.2f z=%.2f", s.x_global, s.y_global, s.z_global)

        cf.land(0.05, config.HLC_LAND_DURATION)
        time.sleep(config.HLC_LAND_DURATION + 0.5)

    log.info("Smoke test done")


if __name__ == "__main__":
    main()
