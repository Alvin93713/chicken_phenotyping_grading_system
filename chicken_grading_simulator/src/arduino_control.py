"""Arduino serial control for external lights and buzzer."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


BAUD_RATE = 115200


@dataclass
class SerialPortInfo:
    device: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.device} - {self.description}" if self.description else self.device


def list_serial_ports() -> List[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []

    return [
        SerialPortInfo(port.device, port.description or "")
        for port in list_ports.comports()
    ]


def send_arduino_commands(port: str, commands: List[str], delay_after_open: float = 0.1) -> List[str]:
    """Open a serial port briefly, send newline-terminated commands, then release it."""
    import serial

    replies: List[str] = []
    with serial.Serial(port, BAUD_RATE, timeout=0.3, write_timeout=0.3) as connection:
        time.sleep(delay_after_open)
        connection.reset_input_buffer()
        for command in commands:
            try:
                connection.write((command.strip() + "\n").encode("utf-8"))
                connection.flush()
                reply = connection.readline().decode("utf-8", errors="replace").strip()
                if reply:
                    replies.append(reply)
            except serial.SerialTimeoutException:
                raise
    return replies


def reconnect_arduino(port: str) -> List[str]:
    """Verify that the Arduino serial port can be opened."""
    return send_arduino_commands(port, ["PING"])


_off_timers: Dict[str, threading.Timer] = {}
_off_timers_lock = threading.Lock()


def light_for(port: str, commands: List[str], seconds: float) -> List[str]:
    """Light for `seconds` without blocking: the Arduino switches off by itself (LIGHT_FOR:<color>:<ms>).
    Firmware without LIGHT_FOR (replies ERR:UNKNOWN:LIGHT_FOR…) gets plain LIGHT commands plus a background
    off-timer. Any earlier off-timer for this port is cancelled first, so it can't cut the new light short."""
    with _off_timers_lock:
        previous = _off_timers.pop(port, None)
    if previous is not None:
        previous.cancel()
    duration_ms = int(seconds * 1000)
    timed = [
        command.replace("LIGHT:", "LIGHT_FOR:", 1) + f":{duration_ms}"
        if command.startswith("LIGHT:") and command != "LIGHT:OFF" else command
        for command in commands
    ]
    replies = send_arduino_commands(port, timed)
    if any(reply.startswith("ERR:UNKNOWN:LIGHT_FOR") for reply in replies):
        replies = send_arduino_commands(port, [command for command in commands if command.startswith("LIGHT:")])

        def turn_off() -> None:
            try:
                send_arduino_commands(port, ["LIGHT:OFF"], delay_after_open=0.02)
            except Exception:
                pass

        timer = threading.Timer(seconds, turn_off)
        timer.daemon = True
        with _off_timers_lock:
            _off_timers[port] = timer
        timer.start()
    return replies


def commands_for_grade(grade_result: Optional[str]) -> List[str]:
    if grade_result == "\u901a\u904e":
        return ["LIGHT:GREEN"]
    if grade_result == "\u5099\u7528":
        return ["LIGHT:YELLOW"]
    if grade_result == "\u4e0d\u901a\u904e":
        return ["LIGHT:RED", "BUZZ:250"]
    return ["LIGHT:OFF"]
