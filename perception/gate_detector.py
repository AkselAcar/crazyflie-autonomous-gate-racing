"""LED-based gate detector for the AI Deck grayscale stream.

Replaces the Webots HSV-magenta detector. The signature now is a thick
bright rectangular RING of LEDs against a dark penumbra — so the
pipeline is:

1. Threshold luminance (LEDs are nearly saturated, background is dim).
2. Morphology: close gaps between adjacent LEDs.
3. findContours with RETR_CCOMP — we use the hierarchy to require that
   each candidate has an internal "hole" (a ring, not a solid blob like
   a backlit window). This is the critical filter that rejects
   img_000400-style false positives.
4. Approximate to 4 corners on the convex hull, reject contours
   touching the image border (partial gates), and apply an aspect-ratio
   sanity check.

Output API matches the Webots `detect_all_gate_contours` (list of
``(corners_4x2, area, contour_idx)``, corners ordered TL/TR/BR/BL) so
the localizer and FSM port stay drop-in.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


def build_mask(image: np.ndarray) -> np.ndarray:
    """Threshold + morphology, exposed for debug visualization."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        image, config.LED_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def detect_all_gate_contours(
    image: np.ndarray,
) -> list[tuple[np.ndarray, float, int]]:
    """Find all LED-ring gates in a grayscale frame.

    Returns a list of ``(corners_4x2, area, contour_idx)``. Corners are
    ordered TL, TR, BR, BL in image space. ``contour_idx`` is the index
    of the original outer contour in the OpenCV hierarchy (kept only for
    debug consistency with the Webots code).
    """
    if image is None:
        return []

    h_img, w_img = image.shape[:2]
    mask = build_mask(image)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # shape (n, 4): [next, prev, first_child, parent]

    results: list[tuple[np.ndarray, float, int]] = []
    border = config.DET_BORDER_MARGIN

    for i, cnt in enumerate(contours):
        # Skip inner ("hole") contours — only iterate outer ones.
        if hierarchy[i][3] != -1:
            continue

        area = cv2.contourArea(cnt)
        if area < config.DET_MIN_AREA:
            continue

        # Require an internal hole — distinguishes a LED ring from a
        # solid bright blob (e.g. backlit window or doorway).
        if hierarchy[i][2] == -1:
            continue

        # Convex-hull + 4-corner approximation suppresses noise on the LED
        # strip (small gaps, individual LED bumps).
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        corners = approx.reshape(4, 2).astype(np.float64)

        # Reject contours touching the image border — these would
        # destroy reprojection / triangulation downstream.
        if any(c[0] <= border or c[0] >= w_img - 1 - border or
               c[1] <= border or c[1] >= h_img - 1 - border
               for c in corners):
            continue

        # Sort TL, TR, BR, BL in image order.
        sorted_y = corners[np.argsort(corners[:, 1])]
        top = sorted_y[:2][np.argsort(sorted_y[:2, 0])]
        bot = sorted_y[2:][np.argsort(sorted_y[2:, 0])]
        ordered = np.vstack([top, bot[::-1]])  # [TL, TR, BR, BL]

        # Aspect-ratio sanity check (gates aren't extreme rectangles).
        w_avg = (np.linalg.norm(ordered[1] - ordered[0]) +
                 np.linalg.norm(ordered[2] - ordered[3])) / 2.0
        h_avg = (np.linalg.norm(ordered[3] - ordered[0]) +
                 np.linalg.norm(ordered[2] - ordered[1])) / 2.0
        if w_avg < 1e-3 or h_avg < 1e-3:
            continue
        aspect = w_avg / h_avg
        if aspect < 0.4 or aspect > 2.5:
            continue

        results.append((ordered, float(area), i))

    return results


_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),     # green
    (255, 0, 0),     # blue (BGR)
    (0, 0, 255),     # red
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
)


def annotate(
    image: np.ndarray,
    detections: list[tuple[np.ndarray, float, int]],
) -> np.ndarray:
    """Return a BGR copy of `image` with each detection overlaid."""
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    for i, (corners, area, _idx) in enumerate(detections):
        col = _PALETTE[i % len(_PALETTE)]
        pts = corners.astype(np.int32)
        cv2.polylines(out, [pts], True, col, 2)
        for p in pts:
            cv2.circle(out, tuple(p), 3, col, -1)
        center = pts.mean(axis=0).astype(np.int32)
        cv2.putText(
            out, f"#{i} A={int(area)}", tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA,
        )
    return out
