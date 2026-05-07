"""Run the gate detector over a folder of images and save side-by-side debug.

Usage (run from C:\\Users\\niels\\Desktop\\Aerial_Robotics):

    python -m Hardware_Project.debug.test_gate_detector \\
        Hardware_Project/stream_out/2026_05_04-14_06_56/raw

Each output image is composed as ``[ original | mask | annotated ]`` so
threshold / morphology / contour issues are easy to spot at a glance.
By default writes to ``<input_dir>/../detected/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from Hardware_Project.perception.gate_detector import (
    annotate,
    build_mask,
    detect_all_gate_contours,
)

log = logging.getLogger(__name__)


def _compose(original: np.ndarray, mask: np.ndarray,
             annotated: np.ndarray) -> np.ndarray:
    """Stack original (grayscale → BGR), mask (BGR), annotated horizontally."""
    left = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    middle = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([left, middle, annotated])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="folder of .png images (e.g. stream_out/.../raw)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output dir (default: <input_dir>/../detected)",
    )
    parser.add_argument(
        "--every", type=int, default=1,
        help="process every Nth image (default: 1)",
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="stop after N images (0 = no limit)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out) if args.out else in_dir.parent / "detected"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))[::args.every]
    if args.max > 0:
        files = files[:args.max]
    log.info("Running detector on %d images, output → %s", len(files), out_dir)

    histogram: dict[int, int] = {}
    for i, p in enumerate(files):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = build_mask(img)
        dets = detect_all_gate_contours(img)
        histogram[len(dets)] = histogram.get(len(dets), 0) + 1
        composed = _compose(img, mask, annotate(img, dets))
        cv2.imwrite(str(out_dir / p.name), composed)
        if (i + 1) % 100 == 0:
            log.info("[%d/%d] processed", i + 1, len(files))

    log.info("Done — detection count distribution:")
    for k in sorted(histogram):
        log.info("    %d gate(s): %d images", k, histogram[k])


if __name__ == "__main__":
    main()
