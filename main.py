"""Top-level entry point: connect, run the FSM, log telemetry.

Usage (from C:\\Users\\niels\\Desktop\\Aerial_Robotics\\Hardware_Project):

    python main.py

Must be launched from inside ``Hardware_Project/`` so that ``config``,
``hardware/``, ``perception/`` and ``planning/`` resolve as top-level
imports. Running ``python -m Hardware_Project.main`` from the parent
directory will fail with ``ModuleNotFoundError: config`` because the
package layout puts ``config.py`` inside ``Hardware_Project/``, not at
the workspace root.

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
# Max age (s) of a camera frame before we treat it as missing. If the AI
# Deck silently stops streaming, ``get_latest()`` keeps returning the
# last frame forever; without this check, the FSM's lost-gate timeout
# never triggers because the same stale image keeps refreshing it.
MAX_FRAME_AGE_S: float = 0.5
# Title of the live FPV window (--display). Reused as the cv2 window handle.
_WINDOW_NAME: str = "Crazyflie FPV"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true",
                        help="run without the AI Deck stream (FSM will not detect gates)")
    parser.add_argument("--camera-transport", choices=("udp", "tcp"),
                        default="udp",
                        help="AI Deck transport: 'tcp' is lossless and "
                             "recommended; 'udp' is the fallback (default)")
    parser.add_argument("--log-dir", default=str(
        Path(__file__).resolve().parent / "logs"))
    parser.add_argument("--save-every", type=int, default=0,
                        help="save every Nth camera frame as "
                             "[original|mask|annotated] composite (0 = off)")
    parser.add_argument("--display", action="store_true",
                        help="show a live window of the drone camera with "
                             "detected gates drawn on top (press 'q' in the "
                             "window to abort the mission)")
    parser.add_argument("--display-mode", choices=("overlay", "composite"),
                        default="overlay",
                        help="'overlay' draws detected gate rectangles on the "
                             "drone view; 'composite' shows "
                             "[original | mask | annotated] side by side "
                             "(default: overlay)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="enable DEBUG logging (visual-servo and "
                             "reprojection traces); default level is INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "trace.csv"
    log.info("Run logs → %s", run_dir)

    with CrazyflieLink() as cf:
        cam_ctx = (AIDeckStream(transport=args.camera_transport)
                   if not args.no_camera else _NullStream())
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
            ticks_with_image = 0
            ticks_without_image = 0
            ticks_stale_image = 0
            last_frame_t: float = -1.0
            frames_dir = run_dir / "frames"
            if args.save_every > 0:
                frames_dir.mkdir(parents=True, exist_ok=True)
            if args.display:
                # WINDOW_NORMAL lets the user resize the (small 324x244) feed.
                cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)

            try:
                while True:
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t += TICK_PERIOD

                    sd = cf.get_state()
                    frame = cam.get_latest()
                    # Drop the image if (a) we already processed this exact
                    # frame on a previous tick (same timestamp), or (b) the
                    # stream stopped delivering new frames recently. Both
                    # symptoms are indistinguishable from the FSM's POV if
                    # we don't filter here.
                    image = None
                    if frame is not None:
                        img, frame_t = frame
                        if frame_t != last_frame_t and (now - frame_t) <= MAX_FRAME_AGE_S:
                            image = img
                            last_frame_t = frame_t
                        else:
                            ticks_stale_image += 1

                    still_running = fsm.tick(sd, image)

                    writer.writerow([
                        f"{now - t_start:.3f}", f"{sd.t:.3f}",
                        fsm.state, fsm.current_gate,
                        f"{sd.x_global:.3f}", f"{sd.y_global:.3f}",
                        f"{sd.z_global:.3f}",
                        f"{sd.roll:.3f}", f"{sd.pitch:.3f}", f"{sd.yaw:.3f}",
                        f"{sd.v_x:.3f}", f"{sd.v_y:.3f}", f"{sd.v_z:.3f}",
                    ])

                    if args.save_every > 0:
                        if image is None:
                            ticks_without_image += 1
                        else:
                            ticks_with_image += 1
                            frame_count += 1
                            if frame_count % args.save_every == 0:
                                try:
                                    _save_debug_frame(image, frames_dir,
                                                      saved_count, fsm)
                                    saved_count += 1
                                except Exception as e:
                                    log.error("Save failed at frame %d: %s",
                                              saved_count, e)

                    if args.display:
                        if image is not None:
                            annotated, composite = _build_views(image, fsm)
                            view = (composite if args.display_mode == "composite"
                                    else annotated)
                            cv2.imshow(_WINDOW_NAME, view)
                        # Pump the GUI event loop every tick (even on a stale
                        # frame) so the window stays responsive and 'q' is
                        # caught promptly.
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            log.warning("Quit requested from display window")
                            break

                    if not still_running:
                        log.info("Mission complete")
                        break
                    if time.monotonic() - t_start > MISSION_TIMEOUT_S:
                        log.error("Mission timeout (%.0fs) — aborting",
                                  MISSION_TIMEOUT_S)
                        break
            except KeyboardInterrupt:
                log.warning("Interrupted by user")
            finally:
                if args.display:
                    cv2.destroyAllWindows()
                if args.save_every > 0:
                    log.info("Image save summary: %d saved / %d ticks with "
                             "image / %d ticks without image / %d ticks "
                             "with stale-or-duplicate image",
                             saved_count, ticks_with_image,
                             ticks_without_image, ticks_stale_image)
            # Whether we exit normally, by timeout, by Ctrl+C or by exception,
            # the CrazyflieLink __exit__ runs the emergency-land procedure
            # before closing the radio link.


def _build_views(image: np.ndarray, fsm: GateFSM,
                 idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run gate detection and build the two BGR debug frames.

    Returns ``(annotated, composite)`` where:
      * ``annotated`` is the drone view with detected gate rectangles drawn
        on top (the "overlay" view).
      * ``composite`` is ``[original | mask | annotated]`` stacked
        horizontally (the "composite" view).

    Both carry an FSM-state caption in the top-left corner; ``idx`` adds a
    ``#NNNNN`` frame-number prefix (used when saving to disk, omitted for the
    live display).
    """
    dets = detect_all_gate_contours(image)
    mask = build_mask(image)
    annotated = annotate(image, dets)
    # np.hstack copies, so captioning ``composite`` below does not bleed into
    # the standalone ``annotated`` frame returned for the overlay view.
    composite = np.hstack([
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        annotated,
    ])
    state_name = fsm._STATE_NAMES.get(fsm.state, "?")
    prefix = f"#{idx:05d} " if idx is not None else ""
    caption = f"{prefix}{state_name} gate={fsm.current_gate} dets={len(dets)}"
    for frame in (annotated, composite):
        cv2.putText(frame, caption, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return annotated, composite


def _save_debug_frame(image: np.ndarray, frames_dir: Path,
                      idx: int, fsm: GateFSM) -> None:
    """Save [original | mask | annotated] composite with FSM state overlay."""
    _, composite = _build_views(image, fsm, idx)
    cv2.imwrite(str(frames_dir / f"img_{idx:05d}.png"), composite)


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
