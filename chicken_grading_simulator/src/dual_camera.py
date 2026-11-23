"""Shared camera readers and calibrated single-image segmentation."""
from __future__ import annotations

import atexit
import base64
import json
import math
import os
import sys
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from inference_utils import BASE_DIR, OUTPUTS_DIR, _mask_sides_px

MODEL_PATHS = {"comb": BASE_DIR / "best_comb.pt", "shank": BASE_DIR / "best_shank.pt"}
ROLES = ("comb", "shank")
PREVIEW_FPS = 30
PREVIEW_JPEG_QUALITY = 80
PREVIEW_PORT = int(os.environ.get("CHICKEN_CAMERA_STREAM_PORT", "8765"))
REALTIME_INFERENCE_FPS = float(os.environ.get("CHICKEN_REALTIME_INFERENCE_FPS", "15"))


class CameraReader:
    def __init__(self, index):
        self.index = index
        self.frame = None
        self.timestamp = 0.0
        self.error = "相機啟動中"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.used = time.monotonic()
        self.started = self.used
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        import cv2
        cap = None
        try:
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f"無法開啟相機 {self.index}")
            while not self.stop.is_set() and time.monotonic() - self.used < 30:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"相機 {self.index} 無法讀取畫面")
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.monotonic()
                    self.error = ""
                self.stop.wait(0.02)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    def latest(self):
        self.used = time.monotonic()
        with self.lock:
            if self.frame is None or time.monotonic() - self.timestamp > 1:
                raise RuntimeError(self.error or f"相機 {self.index} 畫面已過期")
            return self.frame.copy(), self.timestamp


_readers = {}
_reader_lock = threading.Lock()
_preview_server = None
_preview_server_lock = threading.Lock()
_inference_readers = {}
_inference_reader_lock = threading.Lock()
_model_locks = {role: threading.Lock() for role in ROLES}


def reader(index):
    index = int(index)
    with _reader_lock:
        current = _readers.get(index)
        if current is None or (not current.thread.is_alive() and time.monotonic() - current.started >= 5):
            current = _readers[index] = CameraReader(index)
        current.used = time.monotonic()
        return current


def close_cameras():
    with _inference_reader_lock:
        for current in _inference_readers.values():
            current.stop.set()
    with _reader_lock:
        for current in _readers.values():
            current.stop.set()


atexit.register(close_cameras)


def capture_pair(indices, timeout=3.0):
    if indices["comb"] == indices["shank"]:
        raise ValueError("Comb 與 Shank 必須選擇不同相機")
    cameras = {role: reader(indices[role]) for role in ROLES}
    requested = time.monotonic()
    deadline = requested + timeout
    error = "等待兩台相機的新畫面"
    while time.monotonic() < deadline:
        try:
            frames = {role: cameras[role].latest() for role in ROLES}
            times = [item[1] for item in frames.values()]
            if min(times) >= requested and max(times) - min(times) <= 0.25:
                return {role: item[0] for role, item in frames.items()}
        except RuntimeError as exc:
            error = str(exc)
        time.sleep(0.02)
    raise RuntimeError(f"雙相機拍攝失敗：{error}；未建立推論任務")


def encode_preview_jpeg(frame, jpeg_quality=PREVIEW_JPEG_QUALITY):
    """Encode one preview frame for the browser stream."""
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("無法編碼相機預覽畫面")
    return encoded.tobytes()


def preview_data_uri(frame, jpeg_quality=PREVIEW_JPEG_QUALITY):
    """Encode a standalone frame without relying on Streamlit media URLs."""
    payload = base64.b64encode(encode_preview_jpeg(frame, jpeg_quality)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class _PreviewHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _PreviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.strip("/").split("/")
        role = None
        if len(path) == 2 and path[0] == "camera":
            index_part = path[1]
        elif len(path) == 3 and path[0] == "inference" and path[1] in ROLES:
            role = path[1]
            index_part = path[2]
        else:
            self.send_error(404)
            return
        try:
            camera_index = int(index_part.removesuffix(".mjpg"))
            confidence = float(parse_qs(parsed.query).get("confidence", ["0.25"])[0])
        except ValueError:
            self.send_error(400, "Invalid preview parameters")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        interval = 1.0 / PREVIEW_FPS
        try:
            while True:
                started = time.monotonic()
                try:
                    if role is None:
                        frame, _ = reader(camera_index).latest()
                    else:
                        frame, _ = inference_reader(role, camera_index, confidence).latest()
                    encoded = encode_preview_jpeg(frame)
                except RuntimeError:
                    time.sleep(min(interval, 0.05))
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
                self.wfile.write(encoded)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, _format, *_args):
        return


def ensure_preview_server():
    """Start the process-wide MJPEG server and return its TCP port."""
    global _preview_server
    with _preview_server_lock:
        if _preview_server is not None:
            return PREVIEW_PORT
        try:
            server = _PreviewHTTPServer(("0.0.0.0", PREVIEW_PORT), _PreviewHandler)
        except OSError as exc:
            raise RuntimeError(f"無法啟動相機預覽串流 port {PREVIEW_PORT}：{exc}") from exc
        thread = threading.Thread(target=server.serve_forever, name="camera-mjpeg", daemon=True)
        thread.start()
        _preview_server = server
        return PREVIEW_PORT


def save_pair(frames, kind="snapshots"):
    import cv2
    folder = BASE_DIR / "data" / kind / uuid4().hex
    folder.mkdir(parents=True)
    paths = {}
    for role in ROLES:
        path = folder / f"{role}.png"
        # imencode/tofile also supports Windows Unicode paths.
        ok, encoded = cv2.imencode(".png", frames[role])
        if not ok:
            raise RuntimeError(f"無法儲存 {role} 照片")
        encoded.tofile(str(path))
        paths[role] = str(path)
    return paths


@lru_cache(maxsize=4)
def load_model(path, modified):
    from ultralytics import YOLO
    model = YOLO(path)
    if model.task != "segment":
        raise ValueError(f"模型必須是 segmentation：{path}（目前為 {model.task}）")
    return model


def model_for(role):
    path = MODEL_PATHS[role]
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {role} 模型：{path}")
    return load_model(str(path), path.stat().st_mtime_ns)


def target_classes(model, role):
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    items = list(items)
    matches = [int(i) for i, name in items if str(name).strip().lower() == role]
    if not matches and role == "shank" and len(items) == 1:
        matches = [int(items[0][0])]
    if not matches:
        raise ValueError(f"{role} 模型沒有對應類別：{names}")
    return matches


def annotate_preview_frame(role, frame, confidence):
    """Return one YOLO-annotated preview frame without mutating the source frame."""
    import torch

    model = model_for(role)
    classes = target_classes(model, role)
    with _model_locks[role]:
        result = model.predict(
            frame,
            conf=float(confidence),
            classes=classes,
            retina_masks=True,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
    return result.plot()


class InferenceReader:
    """Continuously refresh the latest annotated frame at a bounded rate."""

    def __init__(self, role, index, confidence):
        self.role = role
        self.index = int(index)
        self.confidence = float(confidence)
        self.frame = None
        self.timestamp = 0.0
        self.error = "YOLO 即時預覽啟動中"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.used = time.monotonic()
        self.started = self.used
        self.thread = threading.Thread(
            target=self._run,
            name=f"{role}-camera-{index}-inference",
            daemon=True,
        )
        self.thread.start()

    def set_confidence(self, confidence):
        with self.lock:
            self.confidence = float(confidence)

    def _run(self):
        interval = 1.0 / max(0.1, REALTIME_INFERENCE_FPS)
        while not self.stop.is_set() and time.monotonic() - self.used < 30:
            started = time.monotonic()
            try:
                source, _ = reader(self.index).latest()
                with self.lock:
                    confidence = self.confidence
                annotated = annotate_preview_frame(self.role, source, confidence)
                with self.lock:
                    self.frame = annotated
                    self.timestamp = time.monotonic()
                    self.error = ""
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            remaining = interval - (time.monotonic() - started)
            self.stop.wait(max(0.01, remaining))

    def latest(self):
        self.used = time.monotonic()
        with self.lock:
            if self.frame is None or time.monotonic() - self.timestamp > 3:
                raise RuntimeError(self.error or f"{self.role} YOLO 即時預覽尚未就緒")
            return self.frame.copy(), self.timestamp


def inference_reader(role, index, confidence):
    role = str(role)
    index = int(index)
    if role not in ROLES:
        raise ValueError(f"不支援的推論角色：{role}")
    key = (role, index)
    with _inference_reader_lock:
        current = _inference_readers.get(key)
        if current is None or (
            not current.thread.is_alive() and time.monotonic() - current.started >= 5
        ):
            current = _inference_readers[key] = InferenceReader(role, index, confidence)
        current.used = time.monotonic()
        current.set_confidence(confidence)
        return current


def validate_calibration(calibration, index, shape):
    if not calibration or calibration.get("camera") != index:
        raise ValueError("相機尚未校正，或相機已更換，請重新校正")
    if list(shape[:2]) != calibration.get("shape"):
        raise ValueError("相機解析度與校正照片不同，請重新校正")
    scale = float(calibration["cm_per_pixel"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("校正比例必須為正數")
    return scale


def infer_pair(paths, scales, confidence, progress):
    import cv2
    import numpy as np
    import torch
    folder = OUTPUTS_DIR / Path(paths["comb"]).parent.name
    samples = folder / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    measurements = {}
    metadata = {"source": "dual_camera_snapshot", "sample_frames": [], "cameras": {}}
    for n, role in enumerate(ROLES):
        progress(n, 2, f"正在推論 {role} 單張照片")
        model = model_for(role)
        frame = cv2.imdecode(np.fromfile(paths[role], dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"無法讀取 {role} 照片")
        classes = target_classes(model, role)
        with _model_locks[role]:
            result = model.predict(frame, conf=float(confidence), classes=classes,
                                   retina_masks=True, device=0 if torch.cuda.is_available() else "cpu",
                                   verbose=False)[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            raise ValueError(
                f"{role} 未偵測到有效遮罩（confidence={float(confidence):.2f}），"
                "請降低門檻或更換照片；未更新量測資料"
            )
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != frame.shape[:2]:
            raise ValueError(f"{role} 遮罩與原始照片尺寸不一致")
        mask_areas = [int((candidate > 0.5).sum()) for candidate in masks]
        selected_index = max(range(len(mask_areas)), key=mask_areas.__getitem__)
        mask = masks[selected_index] > 0.5
        confidence_values = []
        confidence_tensor = getattr(result.boxes, "conf", None)
        if confidence_tensor is not None:
            confidence_values = [float(value) for value in confidence_tensor.detach().cpu().tolist()]
        selected_confidence = (
            confidence_values[selected_index]
            if selected_index < len(confidence_values)
            else None
        )
        scale = float(scales[role])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("校正比例必須為正數")
        if role == "comb":
            measurements["comb_area_cm2"] = round(float(mask.sum()) * scale**2, 4)
        else:
            width, length = _mask_sides_px(mask)
            if width is None or length is None:
                raise ValueError("Shank 遮罩無法計算寬度及長度")
            measurements.update(shank_width_cm=round(width * scale, 4),
                                shank_length_cm=round(length * scale, 4))
        sample = samples / f"{role}.png"
        ok, encoded = cv2.imencode(".png", result.plot())
        if not ok:
            raise RuntimeError("無法儲存推論圖片")
        encoded.tofile(str(sample))
        metadata["sample_frames"].append(str(sample))
        metadata["cameras"][role] = {
            "photo": paths[role],
            "cm_per_pixel": scale,
            "model": str(MODEL_PATHS[role]),
            "confidence_threshold": confidence,
            "detection_confidence": selected_confidence,
        }
    metadata["measurements"] = measurements
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(2, 2, "兩張照片推論完成")
    return measurements
