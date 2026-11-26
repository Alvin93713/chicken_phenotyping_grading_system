"""RS485/USB scale serial reader.

The scale packet format is:
STX(0x02) + ASCII grams + ETX(0x03)

Examples:
    100 g:  0x02 0x30 0x31 0x30 0x30 0x03 -> "0100"
    2500 g: 0x02 0x32 0x35 0x30 0x30 0x03 -> "2500"
    Error:  0x02 0x39 0x39 0x39 0x39 0x03 -> "9999"
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from i18n import t


SCALE_BAUD_RATE = 19200
STX = 0x02
ETX = 0x03
ERROR_CODE = "9999"


@dataclass
class ScaleReadResult:
    samples: List[int]
    raw_packets: List[str]
    errors: List[str]

    @property
    def latest_g(self) -> Optional[int]:
        return self.samples[-1] if self.samples else None

    @property
    def average_g(self) -> Optional[float]:
        return statistics.fmean(self.samples) if self.samples else None


def parse_scale_packets(data: bytes) -> ScaleReadResult:
    """Parse all complete STX/ETX packets from a byte buffer."""
    samples: List[int] = []
    raw_packets: List[str] = []
    errors: List[str] = []
    index = 0

    while index < len(data):
        try:
            start = data.index(STX, index)
        except ValueError:
            break
        try:
            end = data.index(ETX, start + 1)
        except ValueError:
            break

        payload_bytes = data[start + 1 : end]
        index = end + 1
        try:
            payload = payload_bytes.decode("ascii").strip()
        except UnicodeDecodeError:
            errors.append(t("非 ASCII 封包：{payload}", payload=repr(payload_bytes)))
            continue

        raw_packets.append(payload)
        if payload == ERROR_CODE:
            errors.append(t("秤重設備回傳異常碼 9999"))
            continue
        if not payload.isdigit():
            errors.append(t("秤重封包不是純數字：{payload}", payload=repr(payload)))
            continue
        samples.append(int(payload))

    return ScaleReadResult(samples=samples, raw_packets=raw_packets, errors=errors)


def split_complete_packets(buffer: bytes) -> tuple[bytes, bytes]:
    """(complete part ending with the last ETX, unfinished rest) — a packet cut between two reads is kept."""
    end = buffer.rfind(bytes([ETX]))
    if end < 0:
        return b"", buffer
    return buffer[: end + 1], buffer[end + 1 :]


@dataclass
class ScaleSnapshot:
    samples: List[tuple]  # (time.monotonic(), grams) within the requested window
    raw_packets: List[str]
    errors: List[str]
    error: str  # connection problem (port missing / busy), "" when reading fine

    @property
    def latest_g(self) -> Optional[int]:
        return self.samples[-1][1] if self.samples else None


class ScaleMonitor:
    """Reads one scale port continuously in a background thread (the port is opened once, no packet is
    missed between reads). The UI only takes the latest readings, so it never waits for the serial port.
    Stops by itself when nobody has asked for readings for IDLE_SECONDS."""

    IDLE_SECONDS = 15.0
    KEEP_SECONDS = 10.0

    def __init__(self, port: str, opener=None):
        self.port = port
        self._opener = opener
        self._lock = threading.Lock()
        self._samples: List[tuple] = []
        self._packets: List[str] = []
        self._errors: List[str] = []
        self._error = ""
        self.used = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"scale-{port}", daemon=True)
        self.thread.start()

    def _open(self):
        if self._opener is not None:
            return self._opener(self.port)
        import serial

        return serial.Serial(self.port, SCALE_BAUD_RATE, timeout=0.1, write_timeout=0.1)

    def _run(self) -> None:
        while not self.stop.is_set() and time.monotonic() - self.used < self.IDLE_SECONDS:
            try:
                with self._open() as connection:  # one monitor per port (scale_monitor): no shared lock
                    with self._lock:
                        self._error = ""
                    pending = b""
                    while not self.stop.is_set() and time.monotonic() - self.used < self.IDLE_SECONDS:
                        chunk = connection.read(getattr(connection, "in_waiting", 0) or 1)
                        if not chunk:
                            continue
                        complete, pending = split_complete_packets(pending + chunk)
                        if complete:
                            self._add(parse_scale_packets(complete))
                        if len(pending) > 256:  # garbage without ETX
                            pending = b""
            except Exception as exc:  # port missing / busy: report it and retry
                with self._lock:
                    self._error = str(exc)
                self.stop.wait(1.0)

    def _add(self, result: ScaleReadResult) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.extend((now, value) for value in result.samples)
            cutoff = now - self.KEEP_SECONDS
            self._samples = [item for item in self._samples if item[0] >= cutoff]
            self._packets = (self._packets + result.raw_packets)[-20:]
            if result.errors:
                self._errors = result.errors[-3:]
            elif result.samples:
                self._errors = []

    def snapshot(self, window_seconds: float = 2.0) -> ScaleSnapshot:
        now = time.monotonic()
        self.used = now
        with self._lock:
            return ScaleSnapshot(
                samples=[item for item in self._samples if item[0] >= now - window_seconds],
                raw_packets=list(self._packets[-5:]),
                errors=list(self._errors),
                error=self._error,
            )


_monitors: dict = {}
_monitors_lock = threading.Lock()


def scale_monitor(port: str) -> ScaleMonitor:
    """The running monitor for `port` (started on first use, restarted if it stopped)."""
    port = str(port or "").strip()
    if not port:
        raise ValueError(t("請先選擇秤重設備 serial port。"))
    with _monitors_lock:
        monitor = _monitors.get(port)
        if monitor is None or not monitor.thread.is_alive():
            monitor = _monitors[port] = ScaleMonitor(port)
        monitor.used = time.monotonic()
        return monitor


def average_recent_samples(history: Iterable[tuple[float, int]]) -> Optional[float]:
    values = [value for _, value in history]
    return statistics.fmean(values) if values else None
