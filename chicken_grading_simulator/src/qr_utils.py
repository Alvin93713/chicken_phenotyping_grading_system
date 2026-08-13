"""QR code 產生與解析工具。"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import qrcode
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
QRCODE_DIR = BASE_DIR / "data" / "qrcodes"


def safe_qrcode_filename(chicken_id):
    """將 chicken_id 轉為可用於檔名的安全字串。"""
    cleaned = str(chicken_id).strip()
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", cleaned)


def generate_qrcode(chicken_id):
    """產生 chicken_id 對應的 QR code 影像。"""
    value = str(chicken_id).strip()
    if not value:
        raise ValueError("chicken_id 不可空白，無法產生 QR code。")

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(value)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def save_qrcode(chicken_id):
    """儲存 QR code 到 qrcodes 資料夾。"""
    QRCODE_DIR.mkdir(parents=True, exist_ok=True)
    image = generate_qrcode(chicken_id)
    path = QRCODE_DIR / f"{safe_qrcode_filename(chicken_id)}.png"
    image.save(path)
    return path


def decode_with_opencv(image_bytes):
    """優先使用 OpenCV 解析 QR code，避免 zbar 安裝問題。"""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return None, f"OpenCV 解析器不可用：{exc}"

    try:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            return None, "OpenCV 無法讀取此圖片格式。"
        detector = cv2.QRCodeDetector()
        value, _, _ = detector.detectAndDecode(image)
    except Exception as exc:  # noqa: BLE001
        return None, f"OpenCV 解析失敗：{exc}"

    value = str(value).strip()
    if not value:
        return None, "OpenCV 未在圖片中偵測到 QR code。"
    return value, None


def decode_with_pyzbar(image_bytes):
    """使用 pyzbar 解析 QR code，作為 OpenCV 備援。"""
    try:
        from pyzbar.pyzbar import decode
    except ImportError as exc:
        return None, f"pyzbar 解析器不可用：{exc}"

    try:
        image = Image.open(BytesIO(image_bytes))
        decoded = decode(image)
    except Exception as exc:  # noqa: BLE001
        return None, f"pyzbar 解析失敗：{exc}"

    if not decoded:
        return None, "pyzbar 未在圖片中偵測到 QR code。"

    value = decoded[0].data.decode("utf-8").strip()
    return value, None


def decode_qrcode_image(file):
    """嘗試解析上傳的 QR code 圖片；OpenCV 失敗時改用 pyzbar。"""
    image_bytes = file.getvalue() if hasattr(file, "getvalue") else file.read()

    value, opencv_error = decode_with_opencv(image_bytes)
    if value:
        return value, None

    value, pyzbar_error = decode_with_pyzbar(image_bytes)
    if value:
        return value, None

    return None, (
        "QR code 解析失敗。請確認圖片清晰且 QR code 完整，或改用手動輸入 chicken_id。"
        f"\nOpenCV：{opencv_error}"
        f"\npyzbar：{pyzbar_error}"
    )
