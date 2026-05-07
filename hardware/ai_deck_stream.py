"""AI Deck JPEG/Bayer streamer.

Receives frames over WiFi (TCP) from the AI Deck JPEG-streamer firmware
and exposes the latest grayscale image to the rest of the pipeline.

Wire format (same as the Bitcraze `opencv-viewer.py` reference):
- 4-byte CPX header: <length:u16, routing:u8, function:u8>
- image header: <magic:0xBC, width:u16, height:u16, depth:u8, format:u8, size:u32>
- payload split into multiple chunks, each prefixed by <length:u16, dst:u8, src:u8>
- format == 0 → raw Bayer (HM01B0 BG pattern); otherwise JPEG

We only need a single grayscale channel for LED detection, so we
demosaic Bayer → grayscale directly (no BGR detour).
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time

import cv2
import numpy as np

from Hardware_Project import config

log = logging.getLogger(__name__)

_HEADER_MAGIC: int = 0xBC


class AIDeckStream:
    """Background thread that reads frames and stores the latest one."""

    def __init__(
        self,
        host: str = config.AI_DECK_IP,
        port: int = config.AI_DECK_PORT,
    ) -> None:
        self._addr = (host, port)
        self._sock: socket.socket | None = None
        self._latest: np.ndarray | None = None
        self._latest_t: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # Lifecycle
    def __enter__(self) -> "AIDeckStream":
        log.info("Connecting to AI Deck %s:%d", *self._addr)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(self._addr)
        self._thread = threading.Thread(
            target=self._loop, name="AIDeckStream", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _rx(self, n: int) -> bytes:
        """Read exactly ``n`` bytes from the socket or raise ConnectionError."""
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("AI Deck socket closed by remote")
            buf.extend(chunk)
        return bytes(buf)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                # Outer CPX header
                hdr = self._rx(4)
                length, _routing, _function = struct.unpack("<HBB", hdr)
                img_hdr = self._rx(length - 2)
                magic, width, height, _depth, fmt, size = struct.unpack(
                    "<BHHBBI", img_hdr,
                )
                if magic != _HEADER_MAGIC:
                    continue

                # Payload comes in arbitrary-sized chunks, each with a small header
                payload = bytearray()
                while len(payload) < size:
                    chunk_hdr = self._rx(4)
                    chunk_len, _dst, _src = struct.unpack("<HBB", chunk_hdr)
                    payload.extend(self._rx(chunk_len - 2))

                if fmt == 0:
                    # Raw Bayer (BG pattern) → demosaic straight to grayscale.
                    bayer = np.frombuffer(bytes(payload), dtype=np.uint8) \
                              .reshape((height, width))
                    gray = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2GRAY)
                else:
                    arr = np.frombuffer(bytes(payload), np.uint8)
                    decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    if decoded is None:
                        continue
                    gray = decoded

                with self._lock:
                    self._latest = gray
                    self._latest_t = time.monotonic()
        except (ConnectionError, OSError) as e:
            if not self._stop.is_set():
                log.error("AI Deck stream stopped: %s", e)

    # Public API
    def get_latest(self) -> tuple[np.ndarray, float] | None:
        """Return ``(grayscale_image, monotonic_timestamp)`` or ``None``."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy(), self._latest_t
