"""Arduino serial control for external lights and buzzer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional


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


def commands_for_grade(grade_result: Optional[str]) -> List[str]:
    if grade_result == "\u901a\u904e":
        return ["LIGHT:GREEN"]
    if grade_result == "\u5099\u7528":
        return ["LIGHT:YELLOW"]
    if grade_result == "\u4e0d\u901a\u904e":
        return ["LIGHT:RED", "BUZZ:250"]
    return ["LIGHT:OFF"]
