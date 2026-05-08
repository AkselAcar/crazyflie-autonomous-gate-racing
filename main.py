"""Top-level entry point: connect, run the FSM, log telemetry.

Usage (from C:\\Users\\niels\\Desktop\\Aerial_Robotics):

    python -m Hardware_Project.main

Spawns:
- The Crazyflie radio link (cflib + Lighthouse log + HLC).
- The AI Deck WiFi camera stream.
- A tick loop that polls state + image, drives the FSM, and writes a
  CSV trace of pose/state for later inspection.

The mission is lap 1 only: take off → 5 gates → land at the take-off
pad. Hit Ctrl-C to abort; the drone will land on disconnect (HLC stop
plus motor cutoff inside cflib).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from hardware.ai_deck_stream import AIDeckStream
from hardware.crazyflie_link import CrazyflieLink
from perception.gate_detector import (
    annotate,
    build_mask,
    detect_all_gate_contours,
)
from planning.fsm import GateFSM

log = logging.getLogger(__name__)

TICK_PERIOD: float = 0.05  # 20 Hz — plenty for HLC re-issue and detection
MISSION_TIMEOUT_S: float = 240.0  # safety: assignment hard cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true",
                        help="run without the AI Deck stream (FSM will not detect gates)")
    parser.add_argument("--log-dir", default=str(
        Path(__file__).resolve().parent / "logs"))
    parser.add_argument("--save-every", type=int, default=0,
                        help="save every Nth camera frame as "
                             "[original|mask|annotated] composite (0 = off)")
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
        cam_ctx = AIDeckStream() if not args.no_camera else _NullStream()
        with cam_ctx as cam, open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_mono", "t_drone", "state", "gate",
                "x", "y", "z", "roll", "pitch", "yaw",
                "vx", "vy", "vz",
            ])

            fsm = GateFSM(cf)
            t_start = time.monotonic()
            next_t = t_start
            frame_count = 0
            saved_count = 0
            frames_dir = run_dir / "frames"
            if args.save_every > 0:
                frames_dir.mkdir(parents=True, exist_ok=True)

            try:
                while True:
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t += TICK_PERIOD

                    sd = cf.get_state()
                    frame = cam.get_latest()
                    image = frame[0] if frame is not None else None

                    still_running = fsm.tick(sd, image)

                    writer.writerow([
                        f"{now - t_start:.3f}", f"{sd.t:.3f}",
                        fsm.state, fsm.current_gate,
                        f"{sd.x_global:.3f}", f"{sd.y_global:.3f}",
                        f"{sd.z_global:.3f}",
                        f"{sd.roll:.3f}", f"{sd.pitch:.3f}", f"{sd.yaw:.3f}",
                        f"{sd.v_x:.3f}", f"{sd.v_y:.3f}", f"{sd.v_z:.3f}",
                    ])

                    if args.save_every > 0 and image is not None:
                        frame_count += 1
                        if frame_count % args.save_every == 0:
                            _save_debug_frame(image, frames_dir, saved_count, fsm)
                            saved_count += 1

                    if not still_running:
                        log.info("Mission complete")
                        break
                    if time.monotonic() - t_start > MISSION_TIMEOUT_S:
                        log.error("Mission timeout (%.0fs) — aborting",
                                  MISSION_TIMEOUT_S)
                        break
            except KeyboardInterrupt:
                log.warning("Interrupted by user")
            # Whether we exit normally, by timeout, by Ctrl+C or by exception,
            # the CrazyflieLink __exit__ runs the emergency-land procedure
            # before closing the radio link.


def _save_debug_frame(image: np.ndarray, frames_dir: Path,
                      idx: int, fsm: GateFSM) -> None:
    """Save [original | mask | annotated] composite with FSM state overlay."""
    dets = detect_all_gate_contours(image)
    mask = build_mask(image)
    composed = np.hstack([
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        annotate(image, dets),
    ])
    state_name = fsm._STATE_NAMES.get(fsm.state, "?")
    cv2.putText(
        composed,
        f"#{idx:05d} {state_name} gate={fsm.current_gate} dets={len(dets)}",
        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(frames_dir / f"img_{idx:05d}.png"), composed)


class _NullStream:
    """Stand-in for AIDeckStream when --no-camera is set."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get_latest(self):
        return None


if __name__ == "__main__":
    main()
