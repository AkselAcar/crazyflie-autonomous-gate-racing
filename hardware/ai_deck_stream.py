"""AI Deck JPEG streamer — UDP or TCP transport.

The AI Deck delivers camera frames as CPX packets. Each CPX packet is a
4-byte header followed by a payload:

  - header: ``<H B B>`` = (length, routing, function)
  - ``length`` counts the 2 routing bytes + payload, so
    ``payload_len = length - 2``
  - an *image header* payload (11 bytes, magic 0xBC):
    ``<B H H B B I>`` = magic, width, height, depth, format, size
    → signals the start of a JPEG; ``size`` data bytes follow in the
    subsequent CPX packets.

Two transports are supported:

  - ``transport="tcp"`` (recommended): the stock Bitcraze
    ``wifi-img-streamer`` protocol. TCP guarantees ordered, lossless
    delivery, so frame assembly is exact — no truncated/overrun frames.
  - ``transport="udp"``: a UDP firmware variant. A ``START_MAGIC`` packet
    kicks off streaming, then frames arrive as a sequence of datagrams.
    UDP has no delivery/order guarantee, so ~half the frames can be lost
    on a noisy WiFi link. Kept as a fallback.

Either way, some JPEGs are truncated a few scan rows short by the GAP8
encoder; ``_decode`` keeps those (still usable) and silences libjpeg's
stderr warning. See ``_suppress_c_stderr``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
_SOI: bytes = b'\xff\xd8'
_EOI: bytes = b'\xff\xd9'
# A buffer that overruns the announced JPEG size by more than this almost
# certainly swallowed the next frame's header datagram (UDP loss).
_OVERRUN_SLACK: int = 2048
# How often the background thread logs its assembly statistics.
_REPORT_INTERVAL_S: float = 5.0
# Socket timeout while reading — short enough to notice a stop request.
_SOCK_TIMEOUT_S: float = 1.0
_CONNECT_TIMEOUT_S: float = 5.0
# One-shot startup debugging: dump the first N CPX headers (UDP only, to
# spot a per-datagram sequence number) and save the first M assembled
# frames to disk so the corruption pattern can be inspected.
_DEBUG_HEADER_DUMP_N: int = 60
_DEBUG_SAVE_N: int = 6


@dataclass
class _StreamStats:
    """Per-window counters for diagnosing frame-assembly health."""

    transport: str = "udp"
    datagrams: int = 0           # datagrams (UDP) / CPX packets (TCP)
    headers_ok: int = 0          # valid image headers
    headers_bad: int = 0         # headers failing the w/h/size check
    orphan_chunks: int = 0       # payload chunks with no active frame
    frames_ok: int = 0           # frames decoded successfully
    frames_truncated: int = 0    # buffer reached size but bad SOI/EOI
    frames_overrun: int = 0      # buffer overran size → missed next header
    frames_preempted: int = 0    # new header arrived before frame completed
    decode_fail: int = 0         # imdecode returned None / wrong shape

    def log(self, window_s: float) -> None:
        decoded = self.frames_ok + self.frames_truncated + self.frames_overrun
        bad = decoded - self.frames_ok
        fps = self.frames_ok / window_s if window_s > 0 else 0.0
        log.info(
            "AI Deck [%s] %.0fs: %.1f fps ok | %d packets, %d frames "
            "(%d ok, %d truncated, %d overrun, %d preempted, %d decode-fail) "
            "| %d bad headers, %d orphan chunks",
            self.transport, window_s, fps, self.datagrams, decoded,
            self.frames_ok, self.frames_truncated, self.frames_overrun,
            self.frames_preempted, self.decode_fail, self.headers_bad,
            self.orphan_chunks,
        )
        if decoded > 0 and bad > 0:
            cause = ("AI Deck encoder-side JPEG truncation" if self.transport
                     == "tcp" else "UDP packet loss")
            log.warning("AI Deck: %d/%d frames discarded as malformed (%s)",
                        bad, decoded, cause)


@contextlib.contextmanager
def _suppress_c_stderr():
    """Temporarily silence C-level stderr (file descriptor 2).

    libjpeg, invoked inside ``cv2.imdecode``, writes "Corrupt JPEG data"
    warnings straight to the process stderr fd — Python's ``logging`` and
    ``warnings`` machinery cannot intercept them. The AI Deck firmware
    ships some JPEGs with a slightly truncated scan (a GAP8 encoder/DMA
    sync issue, not a transport bug); the decoded image is still usable,
    so the warning is pure noise. We redirect fd 2 to the null device for
    the ~1 ms of the decode call only.

    Caveat: this is process-wide, so a log line emitted by another thread
    during that window would be dropped. Acceptable here — decoding is
    brief and the only loss is a stderr log line, never program state.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class AIDeckStream:
    """Background thread that reads camera frames and stores the latest one.

    Args:
        transport: ``"tcp"`` (recommended, lossless) or ``"udp"`` (fallback).
    """

    def __init__(
        self,
        host: str = config.AI_DECK_IP,
        port: int = config.AI_DECK_PORT,
        local_port: int = _LOCAL_PORT,
        transport: str = "udp",
    ) -> None:
        if transport not in ("tcp", "udp"):
            raise ValueError(f"transport must be 'tcp' or 'udp', got {transport!r}")
        self._transport = transport
        self._remote = (host, port)
        self._local_port = local_port
        self._sock: socket.socket | None = None
        self._latest: np.ndarray | None = None
        self._latest_t: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "AIDeckStream":
        if self._transport == "tcp":
            self._open_tcp()
            target = self._loop_tcp
        else:
            self._open_udp()
            target = self._loop_udp
        self._thread = threading.Thread(
            target=target, name="AIDeckStream", daemon=True,
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

    # -- socket setup --------------------------------------------------

    def _open_udp(self) -> None:
        log.info("Opening UDP stream from AI Deck %s:%d (local port %d)",
                 *self._remote, self._local_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUF)
        self._sock.bind(('0.0.0.0', self._local_port))
        self._sock.settimeout(_SOCK_TIMEOUT_S)
        self._sock.sendto(_START_MAGIC, self._remote)

    def _open_tcp(self) -> None:
        log.info("Opening TCP stream from AI Deck %s:%d", *self._remote)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUF)
        sock.settimeout(_CONNECT_TIMEOUT_S)
        try:
            sock.connect(self._remote)
        except OSError as e:
            sock.close()
            raise RuntimeError(
                f"Cannot connect to AI Deck at {self._remote[0]}:"
                f"{self._remote[1]} over TCP ({e}). If the firmware is the "
                f"UDP variant, use transport='udp' instead."
            ) from e
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(_SOCK_TIMEOUT_S)
        self._sock = sock

    # -- TCP receive loop ----------------------------------------------

    def _recv_exact(self, n: int) -> bytearray | None:
        """Read exactly ``n`` bytes from the TCP socket.

        Returns ``None`` if the stream is stopping or the peer closed the
        connection. ``recv`` timeouts are retried so a stop request is
        still noticed promptly.
        """
        assert self._sock is not None
        data = bytearray()
        while len(data) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = self._sock.recv(n - len(data))
            except socket.timeout:
                continue
            if not chunk:
                return None  # peer closed the connection
            data.extend(chunk)
        return data

    def _recv_cpx_packet(self) -> bytearray | None:
        """Read one CPX packet's payload off the TCP stream.

        Header is ``<H B B>`` = (length, routing, function); ``length``
        includes the 2 routing bytes, so ``payload_len = length - 2``.
        """
        header = self._recv_exact(_CPX_HEADER_SIZE)
        if header is None:
            return None
        length = struct.unpack('<H', header[:2])[0]
        if length <= 2:
            return bytearray()
        return self._recv_exact(length - 2)

    def _loop_tcp(self) -> None:
        assert self._sock is not None
        stats = _StreamStats(transport="tcp")
        last_report = time.monotonic()
        dbg_saved = 0

        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_report >= _REPORT_INTERVAL_S:
                    stats.log(now - last_report)
                    stats = _StreamStats(transport="tcp")
                    last_report = now

                payload = self._recv_cpx_packet()
                if payload is None:
                    break  # connection closed or stopping
                stats.datagrams += 1

                # Only image-header packets matter; ignore other CPX traffic.
                if not (len(payload) >= _IMG_HEADER_SIZE
                        and payload[0] == _IMG_HEADER_MAGIC):
                    continue
                _, w, h, _, _, size = struct.unpack(
                    '<BHHBBI', payload[:_IMG_HEADER_SIZE])
                if not (w == config.CAM_WIDTH and h == config.CAM_HEIGHT
                        and 0 < size < _RECV_BUF):
                    stats.headers_bad += 1
                    continue
                stats.headers_ok += 1

                # The JPEG follows as a run of CPX data packets summing
                # to exactly `size` bytes.
                buf = bytearray()
                while len(buf) < size:
                    chunk = self._recv_cpx_packet()
                    if chunk is None:
                        break
                    buf.extend(chunk)
                if len(buf) < size:
                    stats.frames_truncated += 1  # connection dropped mid-frame
                    break

                if dbg_saved < _DEBUG_SAVE_N:
                    ok_soi = buf[:2] == _SOI
                    ok_eoi = buf[size - 2:size] == _EOI
                    out = Path(f"ai_deck_raw_{dbg_saved:02d}.jpg")
                    out.write_bytes(bytes(buf))
                    log.info("DBG frame %d → %s  size=%d buf=%d soi=%s eoi=%s",
                             dbg_saved, out, size, len(buf), ok_soi, ok_eoi)
                    dbg_saved += 1

                gray = self._decode(buf, size, stats)
                if gray is not None:
                    stats.frames_ok += 1
                    with self._lock:
                        self._latest = gray
                        self._latest_t = time.monotonic()

        except OSError as e:
            if not self._stop.is_set():
                log.error("AI Deck TCP stream stopped: %s", e)

    # -- UDP receive loop ----------------------------------------------

    def _loop_udp(self) -> None:
        assert self._sock is not None
        buf = bytearray()
        expected_size = 0
        receiving = False

        stats = _StreamStats(transport="udp")
        last_report = time.monotonic()

        # One-shot startup debugging (see _DEBUG_* constants).
        dbg_headers = 0   # CPX headers dumped so far
        dbg_saved = 0     # raw frames written to disk so far
        chunks = 0        # payload chunks in the frame currently assembling

        try:
            while not self._stop.is_set():
                # Periodic health report, runs even while idle (timeout=1s).
                now = time.monotonic()
                if now - last_report >= _REPORT_INTERVAL_S:
                    stats.log(now - last_report)
                    stats = _StreamStats(transport="udp")
                    last_report = now

                try:
                    data, _ = self._sock.recvfrom(2048)
                except socket.timeout:
                    continue

                stats.datagrams += 1
                if len(data) < _CPX_HEADER_SIZE:
                    continue
                payload = data[_CPX_HEADER_SIZE:]

                # Dump the raw CPX header of the first N datagrams. If it
                # carries a per-datagram counter, one of these 4 bytes will
                # increment monotonically across consecutive lines — that
                # would be the sequence number we'd use to reorder.
                if dbg_headers < _DEBUG_HEADER_DUMP_N:
                    dbg_headers += 1
                    is_hdr = (len(payload) >= _IMG_HEADER_SIZE
                              and payload[0] == _IMG_HEADER_MAGIC)
                    log.info("DBG dg#%-3d cpx=%s payload=%4dB %s",
                             dbg_headers, data[:_CPX_HEADER_SIZE].hex(),
                             len(payload), "<IMG-HEADER>" if is_hdr
                             else f"data[{payload[:4].hex()}..]")

                # Image header → start of new frame.
                if (len(payload) >= _IMG_HEADER_SIZE
                        and payload[0] == _IMG_HEADER_MAGIC):
                    _, w, h, _, _, size = struct.unpack(
                        '<BHHBBI', payload[:_IMG_HEADER_SIZE])
                    if receiving:
                        # Previous frame never reached expected_size.
                        stats.frames_preempted += 1
                    if (w == config.CAM_WIDTH and h == config.CAM_HEIGHT
                            and 0 < size < 65536):
                        stats.headers_ok += 1
                        expected_size = size
                        buf = bytearray()
                        receiving = True
                        chunks = 0
                    else:
                        # Bad header: drop state so the new frame's chunks
                        # don't pile onto a stale buffer.
                        stats.headers_bad += 1
                        receiving = False
                        buf = bytearray()
                    continue

                if not receiving:
                    stats.orphan_chunks += 1
                    continue

                buf.extend(payload)
                chunks += 1

                # Desync guard: a wide overrun means we missed the next
                # frame's header and are now concatenating two frames.
                if len(buf) > expected_size + _OVERRUN_SLACK:
                    stats.frames_overrun += 1
                    receiving = False
                    buf = bytearray()
                    continue

                if len(buf) >= expected_size:
                    # Save the first few assembled buffers verbatim so the
                    # corruption can be inspected (open the .jpg files).
                    if dbg_saved < _DEBUG_SAVE_N:
                        ok_soi = buf[:2] == _SOI
                        ok_eoi = buf[expected_size - 2:expected_size] == _EOI
                        out = Path(f"ai_deck_raw_{dbg_saved:02d}.jpg")
                        out.write_bytes(bytes(buf))
                        log.info("DBG frame %d → %s  size=%d buf=%d chunks=%d "
                                 "soi=%s eoi=%s", dbg_saved, out, expected_size,
                                 len(buf), chunks, ok_soi, ok_eoi)
                        dbg_saved += 1
                    gray = self._decode(buf, expected_size, stats)
                    if gray is not None:
                        stats.frames_ok += 1
                        with self._lock:
                            self._latest = gray
                            self._latest_t = time.monotonic()
                    receiving = False
                    buf = bytearray()

        except OSError as e:
            if not self._stop.is_set():
                log.error("AI Deck stream stopped: %s", e)

    # -- decoding ------------------------------------------------------

    def _decode(self, buf: bytearray, expected_size: int,
                stats: _StreamStats) -> np.ndarray | None:
        """Validate and decode a fully-assembled frame buffer.

        A correctly assembled CPX frame is exactly ``expected_size`` bytes,
        starts with the JPEG SOI marker and ends with EOI. If either marker
        is missing the buffer is misassembled (UDP loss) — we discard it
        here rather than hand a truncated JPEG to ``cv2.imdecode``, which is
        what triggers libjpeg's "premature end of data segment" spam.
        """
        if expected_size < _MIN_JPEG_BYTES:
            stats.frames_truncated += 1
            return None
        if (buf[:2] != _SOI
                or buf[expected_size - 2:expected_size] != _EOI):
            stats.frames_truncated += 1
            return None

        arr = np.frombuffer(buf, np.uint8, count=expected_size)
        # The AI Deck occasionally truncates the JPEG scan a few rows
        # short; libjpeg warns on stderr but still returns a usable image.
        with _suppress_c_stderr():
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[:2] != (config.CAM_HEIGHT, config.CAM_WIDTH):
            stats.decode_fail += 1
            return None
        return img

    # -- public API ----------------------------------------------------

    def get_latest(self) -> tuple[np.ndarray, float] | None:
        """Return ``(grayscale_image, monotonic_timestamp)`` or ``None``."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy(), self._latest_t
