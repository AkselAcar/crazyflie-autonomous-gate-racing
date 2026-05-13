"""AI Deck JPEG streamer over UDP.

Matches the wire protocol used by the Bitcraze FPV example (UDP, not TCP).
The AI Deck sends a `START_MAGIC` packet to begin streaming, then delivers
frames as a sequence of UDP datagrams, each prefixed by a 4-byte CPX header.

Wire format (per datagram):
  - 4-byte CPX header (discarded here)
  - image header (11 bytes, magic 0xBC): <B H H B B I> = magic, w, h, ?, ?, size
    → signals start of a new JPEG frame
  - subsequent datagrams carry raw payload bytes to fill `size` bytes total
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

_CPX_HEADER_SIZE: int = 4
_IMG_HEADER_MAGIC: int = 0xBC
_IMG_HEADER_SIZE: int = 11
_MIN_JPEG_BYTES: int = 5000
_START_MAGIC: bytes = b'FER'
_LOCAL_PORT: int = 5001
_RECV_BUF: int = 1 << 20


class AIDeckStream:
    """Background thread that reads UDP frames and stores the latest one."""

    def __init__(
        self,
        host: str = config.AI_DECK_IP,
        port: int = config.AI_DECK_PORT,
        local_port: int = _LOCAL_PORT,
    ) -> None:
        self._remote = (host, port)
        self._local_port = local_port
        self._sock: socket.socket | None = None
        self._latest: np.ndarray | None = None
        self._latest_t: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "AIDeckStream":
        log.info("Opening UDP stream from AI Deck %s:%d (local port %d)",
                 *self._remote, self._local_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUF)
        self._sock.bind(('0.0.0.0', self._local_port))
        self._sock.sendto(_START_MAGIC, self._remote)
        self._thread = threading.Thread(
            target=self._loop, name="AIDeckStream", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        assert self._sock is not None
        buf = bytearray()
        expected_size = 0
        receiving = False

        try:
            while not self._stop.is_set():
                self._sock.settimeout(1.0)
                try:
                    data, _ = self._sock.recvfrom(2048)
                except socket.timeout:
                    continue

                if len(data) < _CPX_HEADER_SIZE:
                    continue
                payload = data[_CPX_HEADER_SIZE:]

                # Image header → start of new frame
                if len(payload) >= _IMG_HEADER_SIZE and payload[0] == _IMG_HEADER_MAGIC:
                    _, w, h, _, _, size = struct.unpack(
                        '<BHHBBI', payload[:_IMG_HEADER_SIZE])
                    if (w == config.CAM_WIDTH and h == config.CAM_HEIGHT
                            and 0 < size < 65536):
                        expected_size = size
                        buf = bytearray()
                        receiving = True
                    continue

                if not receiving:
                    continue

                buf.extend(payload)

                if len(buf) >= expected_size:
                    gray = self._decode(buf)
                    if gray is not None:
                        with self._lock:
                            self._latest = gray
                            self._latest_t = time.monotonic()
                    receiving = False

        except OSError as e:
            if not self._stop.is_set():
                log.error("AI Deck stream stopped: %s", e)

    def _decode(self, buf: bytearray) -> np.ndarray | None:
        soi = buf.find(b'\xff\xd8')
        eoi = buf.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return None
        jpeg_len = eoi + 2 - soi
        if jpeg_len < _MIN_JPEG_BYTES:
            return None
        arr = np.frombuffer(buf, np.uint8, count=jpeg_len, offset=soi)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[:2] != (config.CAM_HEIGHT, config.CAM_WIDTH):
            return None
        return img

    # Public API
    def get_latest(self) -> tuple[np.ndarray, float] | None:
        """Return ``(grayscale_image, monotonic_timestamp)`` or ``None``."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy(), self._latest_t
