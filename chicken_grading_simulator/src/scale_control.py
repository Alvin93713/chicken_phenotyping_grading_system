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
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


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
            errors.append(f"非 ASCII 封包：{payload_bytes!r}")
            continue

        raw_packets.append(payload)
        if payload == ERROR_CODE:
            errors.append("秤重設備回傳異常碼 9999")
            continue
        if not payload.isdigit():
            errors.append(f"秤重封包不是純數字：{payload!r}")
            continue
        samples.append(int(payload))

    return ScaleReadResult(samples=samples, raw_packets=raw_packets, errors=errors)


def _read_serial_bytes(port: str, duration_seconds: float, timeout: float = 0.1) -> bytes:
    import serial

    chunks: List[bytes] = []
    deadline = time.monotonic() + max(0.05, float(duration_seconds))
    with serial.Serial(
        port,
        SCALE_BAUD_RATE,
        timeout=timeout,
        write_timeout=timeout,
    ) as connection:
        connection.reset_input_buffer()
        while time.monotonic() < deadline:
            waiting = connection.in_waiting
            chunk = connection.read(waiting or 1)
            if chunk:
                chunks.append(chunk)
    return b"".join(chunks)


def read_scale_samples(port: str, duration_seconds: float = 0.5) -> ScaleReadResult:
    """Read scale packets for a short window and return parsed gram samples."""
    if not str(port or "").strip():
        raise ValueError("請先選擇秤重設備 serial port。")
    data = _read_serial_bytes(str(port).strip(), duration_seconds)
    return parse_scale_packets(data)


def read_scale_average(port: str, duration_seconds: float = 2.0) -> ScaleReadResult:
    """Read scale packets and calculate average grams over the requested window."""
    return read_scale_samples(port, duration_seconds=duration_seconds)


def append_recent_scale_samples(
    history: Iterable[tuple[float, int]],
    samples: Iterable[int],
    window_seconds: float = 2.0,
) -> list[tuple[float, int]]:
    """Append samples with timestamps and keep only a recent time window."""
    now = time.monotonic()
    cutoff = now - float(window_seconds)
    merged = [(timestamp, value) for timestamp, value in history if timestamp >= cutoff]
    merged.extend((now, int(value)) for value in samples)
    return merged


def average_recent_samples(history: Iterable[tuple[float, int]]) -> Optional[float]:
    values = [value for _, value in history]
    return statistics.fmean(values) if values else None
