"""Synchronously record AI Deck images and Lighthouse poses.

Usage (run from C:\\Users\\niels\\Desktop\\Aerial_Robotics\\Hardware_Project):

    python debug/image_capture.py --duration 30 --rate 5

The drone is NOT controlled by this script — fly it manually with
cfclient (or hand-walk it). Each captured frame is saved as PNG and one
row is appended to ``poses.csv`` with the corresponding pose at capture
time. The two streams (radio + WiFi) run concurrently, so the pose and
the frame are within a few ms of each other.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Allow running this script directly from Hardware_Project/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.ai_deck_stream import AIDeckStream
from hardware.crazyflie_link import CrazyflieLink

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0,
                        help="recording duration in seconds")
    parser.add_argument("--rate", type=float, default=5.0,
                        help="capture rate in Hz")
    default_out = Path(__file__).resolve().parent.parent / "captures"
    parser.add_argument("--out", default=str(default_out),
                        help="output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / stamp
    img_dir = run_dir / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "poses.csv"

    period = 1.0 / args.rate
    n_saved = 0

    with CrazyflieLink() as cf, AIDeckStream() as cam, \
            open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "t", "x", "y", "z", "roll", "pitch", "yaw",
        ])

        log.info("Recording %.1fs at %.1f Hz to %s",
                 args.duration, args.rate, run_dir)
        t_start = time.monotonic()
        next_t = t_start

        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            frame = cam.get_latest()
            if frame is None:
                continue
            img, _ = frame
            state = cf.get_state()

            fname = f"frame_{n_saved:06d}.png"
            cv2.imwrite(str(img_dir / fname), img)
            writer.writerow([
                n_saved,
                f"{state.t:.4f}",
                f"{state.x_global:.4f}",
                f"{state.y_global:.4f}",
                f"{state.z_global:.4f}",
                f"{state.roll:.4f}",
                f"{state.pitch:.4f}",
                f"{state.yaw:.4f}",
            ])
            n_saved += 1

        log.info("Saved %d frames to %s", n_saved, run_dir)


if __name__ == "__main__":
    main()
