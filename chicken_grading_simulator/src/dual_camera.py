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
from contextlib import contextmanager
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from i18n import t
from inference_utils import BASE_DIR, OUTPUTS_DIR, _mask_sides_px
import model_registry
import product_review
import training_labels

# The model version in use per role (models/best_<role>_<雞舍>_<日期>.pt, see model_registry); falls back
# to best_<role>.pt in the project folder when no version has been imported.
MODEL_PATHS = model_registry.ActiveModels()
ROLES = ("comb", "shank")
PREVIEW_FPS = 30
PREVIEW_JPEG_QUALITY = 80
PREVIEW_PORT = int(os.environ.get("CHICKEN_CAMERA_STREAM_PORT", "8765"))
REALTIME_INFERENCE_FPS = float(os.environ.get("CHICKEN_REALTIME_INFERENCE_FPS", "15"))
INFERENCE_IDLE_SECONDS = 3.0


class _CameraError(RuntimeError):
    """Reader-thread error kept as a template so it is translated when shown in the UI."""

    def __init__(self, template, **values):
        super().__init__(template.format(**values))
        self.template = template
        self.values = values


class CameraReader:
    def __init__(self, index):
        self.index = index
        self.frame = None
        self.timestamp = 0.0
        # (template, values); formatted with t() in latest(), which runs in the caller's thread.
        self.error = ("相機啟動中", {})
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
                raise _CameraError("無法開啟相機 {index}", index=self.index)
            while not self.stop.is_set() and time.monotonic() - self.used < 30:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise _CameraError("相機 {index} 無法讀取畫面", index=self.index)
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.monotonic()
                    self.error = None
                self.stop.wait(0.02)
        except Exception as exc:
            with self.lock:
                if isinstance(exc, _CameraError):
                    self.error = (exc.template, exc.values)
                else:
                    self.error = (str(exc), {})
        finally:
            if cap is not None:
                cap.release()

    def latest(self):
        self.used = time.monotonic()
        with self.lock:
            if self.frame is None or time.monotonic() - self.timestamp > 1:
                template, values = self.error or ("相機 {index} 畫面已過期", {"index": self.index})
                raise RuntimeError(t(template, **values))
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
        raise ValueError(t("Comb 與 Shank 必須選擇不同相機"))
    cameras = {role: reader(indices[role]) for role in ROLES}
    requested = time.monotonic()
    deadline = requested + timeout
    error = t("等待兩台相機的新畫面")
    while time.monotonic() < deadline:
        try:
            frames = {role: cameras[role].latest() for role in ROLES}
            times = [item[1] for item in frames.values()]
            if min(times) >= requested and max(times) - min(times) <= 0.25:
                return {role: item[0] for role, item in frames.items()}
        except RuntimeError as exc:
            error = str(exc)
        time.sleep(0.02)
    raise RuntimeError(t("雙相機拍攝失敗：{error}；未建立推論任務", error=error))


def capture_one(index, timeout=3.0):
    """One fresh frame from one camera (used to calibrate each camera on its own)."""
    camera = reader(index)
    requested = time.monotonic()
    deadline = requested + timeout
    error = t("等待相機的新畫面")
    while time.monotonic() < deadline:
        try:
            frame, stamp = camera.latest()
            if stamp >= requested:
                return frame
        except RuntimeError as exc:
            error = str(exc)
        time.sleep(0.02)
    raise RuntimeError(t("相機 {index} 拍攝失敗：{error}", index=index, error=error))


def save_one(frame, role, kind="calibration_captures"):
    """Save one camera's photo as data/<kind>/<id>/<role>.png and return its path."""
    import cv2
    folder = BASE_DIR / "data" / kind / uuid4().hex
    folder.mkdir(parents=True)
    path = folder / f"{role}.png"
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(t("無法儲存 {role} 照片", role=t(ROLE_ZH.get(role, role))))
    encoded.tofile(str(path))
    return str(path)


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
            confidence = float(parse_qs(parsed.query).get("confidence", ["0.5"])[0])
        except ValueError:
            self.send_error(400, "Invalid preview parameters")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        interval = 1.0 / PREVIEW_FPS
        last_stamp = None
        try:
            while True:
                started = time.monotonic()
                try:
                    if role is None:
                        frame, stamp = reader(camera_index).latest()
                    else:
                        frame, stamp = inference_reader(role, camera_index, confidence).latest()
                    if stamp == last_stamp:  # nothing new (inference runs at 15 fps): don't re-encode
                        time.sleep(min(interval, 0.02))
                        continue
                    last_stamp = stamp
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
            raise RuntimeError(t("無法啟動相機預覽串流 port {port}：{error}",
                                 port=PREVIEW_PORT, error=exc)) from exc
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
            raise RuntimeError(t("無法儲存 {role} 照片", role=t(ROLE_ZH.get(role, role))))
        encoded.tofile(str(path))
        paths[role] = str(path)
    return paths


@lru_cache(maxsize=4)
def load_model(path, modified):
    from ultralytics import YOLO
    # A TensorRT engine is given its task explicitly (its metadata may not carry it).
    model = YOLO(path, task="segment") if str(path).endswith(".engine") else YOLO(path)
    if model.task != "segment":
        raise ValueError(f"模型必須是 segmentation：{path}（目前為 {model.task}）")
    return model


# --- TensorRT first, PyTorch fallback ------------------------------------------------------------------------
# On the Orin an FP16 engine is built next to each model: best_comb.pt -> best_comb.engine (same folder, same
# name). The engine is used when it exists and is not older than its .pt (a newer .pt means the engine was
# built from the previous model); otherwise, or when the engine cannot be loaded or run, the .pt is used.
# CHICKEN_DISABLE_TENSORRT=1 always uses the .pt.
RUNTIME_TENSORRT, RUNTIME_PYTORCH = "TensorRT", "PyTorch"
_engine_failures: dict[str, str] = {}  # engine path -> error; a failed engine is not tried again until it changes


def _engine_key(engine: Path) -> str:
    return f"{engine}|{engine.stat().st_mtime_ns}"  # a rebuilt engine (new mtime) is tried again


def engine_path(role) -> Path:
    return Path(MODEL_PATHS[role]).with_suffix(".engine")


def runtime_model(role) -> tuple[Path, str]:
    """(file to load, runtime) for `role`."""
    pt = Path(MODEL_PATHS[role])
    engine = pt.with_suffix(".engine")
    if os.environ.get("CHICKEN_DISABLE_TENSORRT") != "1" and engine.is_file() \
            and _engine_key(engine) not in _engine_failures \
            and (not pt.is_file() or engine.stat().st_mtime_ns >= pt.stat().st_mtime_ns):
        return engine, RUNTIME_TENSORRT
    return pt, RUNTIME_PYTORCH


def runtime_status(role) -> dict:
    """What the UI shows: the runtime in use and, when the engine is not used, why."""
    pt = Path(MODEL_PATHS[role])
    engine = pt.with_suffix(".engine")
    path, runtime = runtime_model(role)
    reason = ""
    if runtime == RUNTIME_PYTORCH and engine.is_file():
        if os.environ.get("CHICKEN_DISABLE_TENSORRT") == "1":
            reason = "已停用 TensorRT"
        elif _engine_key(engine) in _engine_failures:
            reason = "TensorRT 載入或推論失敗，已改用 PT"
        else:
            reason = "TensorRT 檔比 PT 舊，等待重建"
    return {"runtime": runtime, "path": str(path), "reason": reason}


def _engine_failed(engine: Path, exc: Exception) -> None:
    _engine_failures[_engine_key(engine)] = str(exc)
    print(f"TensorRT engine failed, falling back to PT: {engine}: {exc}", file=sys.stderr)


def model_for(role):
    path, runtime = runtime_model(role)
    if not path.is_file():
        raise FileNotFoundError(f"找不到{ROLE_ZH.get(role, role)}模型：{path}")
    try:
        return load_model(str(path), path.stat().st_mtime_ns)
    except Exception as exc:
        pt = Path(MODEL_PATHS[role])
        if runtime != RUNTIME_TENSORRT or not pt.is_file():
            raise
        _engine_failed(path, exc)
        return load_model(str(pt), pt.stat().st_mtime_ns)


# --- snapshot jobs come first ---------------------------------------------------------------------------------
# While a snapshot job is inferring, the live previews show the plain camera picture instead of running YOLO on
# every frame: preview inference (2 cameras × 15 fps) shares the GPU / Python with the job and made it several
# times slower. The previews return to inference as soon as no job is running.
_snapshot_jobs = 0
_snapshot_jobs_lock = threading.Lock()
_no_snapshot_job = threading.Event()
_no_snapshot_job.set()


@contextmanager
def snapshot_inference():
    """Mark a running snapshot job (nesting and parallel jobs are counted)."""
    global _snapshot_jobs
    with _snapshot_jobs_lock:
        _snapshot_jobs += 1
        _no_snapshot_job.clear()
    try:
        yield
    finally:
        with _snapshot_jobs_lock:
            _snapshot_jobs -= 1
            if _snapshot_jobs == 0:
                _no_snapshot_job.set()


def preview_inference_paused() -> bool:
    return not _no_snapshot_job.is_set()


# --- load (and warm up) both models when the app starts -------------------------------------------------------
# The first prediction of a TensorRT engine (or .pt) loads it and builds its CUDA context: ~12 s on the Orin,
# which the first chicken used to wait for. A background thread does it at start-up instead.
# CHICKEN_PRELOAD_MODELS=0 turns it off (tests).
_preload_lock = threading.Lock()
_preload_started = False
preload_state: dict[str, str] = {}  # role -> "loading" / "ready" / "error: …"


def _warm_up(role):
    import numpy as np

    preload_state[role] = "loading"
    try:
        predict_role(role, np.zeros((720, 1280, 3), dtype=np.uint8), 0.5)  # loads the model and runs it once
        preload_state[role] = "ready"
    except Exception as exc:  # a missing / broken model is reported when it is really used
        preload_state[role] = f"error: {exc}"
        print(f"Model preload failed for {role}: {exc}", file=sys.stderr)


def preload_models() -> bool:
    """Start loading + warming both models in the background, once per process. Returns True when started."""
    global _preload_started
    if os.environ.get("CHICKEN_PRELOAD_MODELS", "1") == "0":
        return False
    with _preload_lock:
        if _preload_started:
            return False
        _preload_started = True
    threading.Thread(target=lambda: [_warm_up(role) for role in ROLES], name="model-preload", daemon=True).start()
    return True


def predict_role(role, frame, confidence):
    """One segmentation prediction for `role` (engine first; if the engine fails while predicting, the same
    frame is predicted again with the .pt). Returns (result, model)."""
    import torch

    for attempt in range(2):
        path, runtime = runtime_model(role)
        model = model_for(role)
        classes = target_classes(model, role)
        try:
            with _model_locks[role]:
                result = model.predict(frame, conf=float(confidence), classes=classes, retina_masks=True,
                                       device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
            return result, model
        except Exception as exc:
            if attempt or runtime != RUNTIME_TENSORRT or not Path(MODEL_PATHS[role]).is_file():
                raise
            _engine_failed(path, exc)


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
    result, _ = predict_role(role, frame, confidence)
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
        # Stop ~3 s after the last viewer: preview inference (2 cameras × 15 fps) competes with the page for
        # the CPU / Python, and used to keep running for 30 s after leaving the page, slowing page switches.
        while not self.stop.is_set() and time.monotonic() - self.used < INFERENCE_IDLE_SECONDS:
            started = time.monotonic()
            try:
                source, _ = reader(self.index).latest()
                with self.lock:
                    confidence = self.confidence
                if preview_inference_paused():  # a snapshot job is inferring: plain picture, no YOLO
                    annotated = source
                else:
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
        raise ValueError(t("相機尚未校正，或相機已更換，請重新校正"))
    if list(shape[:2]) != calibration.get("shape"):
        raise ValueError(t("相機解析度與校正照片不同，請重新校正"))
    scale = float(calibration["cm_per_pixel"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(t("校正比例必須為正數"))
    return scale


MASK_CONFIDENCE_MARGIN = 0.15
ROLE_ZH = {"comb": "雞冠", "shank": "腳脛"}  # role names in messages shown to the user


def select_mask_index(areas, confidences, margin=MASK_CONFIDENCE_MARGIN):
    """Pick the detection to measure using both confidence and size: among detections whose confidence is
    within `margin` of the most confident one, take the largest mask. A big but doubtful detection (e.g.
    background) can no longer beat a confident one; a tiny fragment can't beat the real object."""
    indices = range(len(areas))
    if len(confidences) == len(areas) and confidences:
        best = max(confidences)
        indices = [i for i in indices if confidences[i] >= best - margin]
    return max(indices, key=lambda i: areas[i])


LEG_AGREEMENT_TOLERANCE = 0.15  # two legs within this relative difference → averaged (testing 15%; plan 10%)
SECOND_LEG_MIN_AREA_RATIO = 0.25  # a smaller mask counts as the other leg only if at least this share of the first
SECOND_LEG_MAX_OVERLAP = 0.5  # …and it is not the same leg detected twice (overlap / smaller mask)


def select_leg_indices(areas, confidences, overlaps=None, margin=MASK_CONFIDENCE_MARGIN):
    """Up to two shank detections, one per leg: the mask select_mask_index picks, plus the next largest
    confident mask that is big enough and does not cover the same leg. overlaps(i, j) → intersection / smaller."""
    first = select_mask_index(areas, confidences, margin)
    candidates = range(len(areas))
    if len(confidences) == len(areas) and confidences:
        best = max(confidences)
        candidates = [i for i in candidates if confidences[i] >= best - margin]
    for i in sorted(candidates, key=lambda i: areas[i], reverse=True):
        if i == first or areas[i] < SECOND_LEG_MIN_AREA_RATIO * areas[first]:
            continue
        if overlaps is not None and overlaps(first, i) > SECOND_LEG_MAX_OVERLAP:
            continue
        return [first, i]
    return [first]


def combine_legs(legs, tolerance=LEG_AGREEMENT_TOLERANCE):
    """legs: [{"width", "length"}] in cm, the first one from the largest mask.
    One leg → its values. Two legs agreeing within `tolerance` on width and length → their mean.
    Otherwise → the largest-mask leg (the other is most likely occluded or tilted) and `disagree` is set,
    which sends the case to review. Returns (width, length, info)."""
    if len(legs) < 2:
        return legs[0]["width"], legs[0]["length"], {"method": "single", "legs": legs, "disagree": False}
    a, b = legs[:2]
    differences = {key: abs(a[key] - b[key]) / max(a[key], b[key]) for key in ("width", "length")}
    info = {"legs": legs[:2], "difference": {k: round(v, 4) for k, v in differences.items()},
            "tolerance": tolerance}
    if max(differences.values()) <= tolerance:
        return (a["width"] + b["width"]) / 2, (a["length"] + b["length"]) / 2, dict(info, method="mean", disagree=False)
    return a["width"], a["length"], dict(info, method="largest", disagree=True)


def infer_pair(paths, scales, confidence, progress):
    """Infer one snapshot (comb + shank photos); live preview inference pauses meanwhile."""
    with snapshot_inference():
        return _infer_pair(paths, scales, confidence, progress)


def _infer_pair(paths, scales, confidence, progress):
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
        frame = cv2.imdecode(np.fromfile(paths[role], dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"無法讀取{ROLE_ZH[role]}照片")
        result, model = predict_role(role, frame, confidence)
        runtime = runtime_model(role)
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            raise ValueError(
                f"{ROLE_ZH[role]}未偵測到有效遮罩（信心門檻 {float(confidence):.2f}），"
                "請降低門檻或更換照片；未更新量測資料"
            )
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != frame.shape[:2]:
            raise ValueError(f"{role} 遮罩與原始照片尺寸不一致")
        mask_areas = [int((candidate > 0.5).sum()) for candidate in masks]
        confidence_values = []
        confidence_tensor = getattr(result.boxes, "conf", None)
        if confidence_tensor is not None:
            confidence_values = [float(value) for value in confidence_tensor.detach().cpu().tolist()]
        selected_index = select_mask_index(mask_areas, confidence_values)
        mask = masks[selected_index] > 0.5
        selected_confidence = (
            confidence_values[selected_index]
            if selected_index < len(confidence_values)
            else None
        )
        scale = float(scales[role])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("校正比例必須為正數")
        leg_check = None
        if role == "comb":
            measurements["comb_area_cm2"] = round(float(mask.sum()) * scale**2, 4)
        else:
            binary = masks > 0.5
            overlaps = lambda i, j: float((binary[i] & binary[j]).sum()) / max(1, min(mask_areas[i], mask_areas[j]))
            legs = []
            for index in select_leg_indices(mask_areas, confidence_values, overlaps):
                width, length = _mask_sides_px(binary[index])
                if width is not None and length is not None:
                    legs.append({"width": width * scale, "length": length * scale,
                                 "confidence": confidence_values[index] if index < len(confidence_values) else None})
            if not legs:
                raise ValueError("腳脛遮罩無法計算寬度及長度")
            width_cm, length_cm, leg_check = combine_legs(legs)
            measurements.update(shank_width_cm=round(width_cm, 4), shank_length_cm=round(length_cm, 4))
        sample = samples / f"{role}.png"
        ok, encoded = cv2.imencode(".png", result.plot())
        if not ok:
            raise RuntimeError("無法儲存推論圖片")
        encoded.tofile(str(sample))
        # Keep the masks shown on the result image as YOLO-seg labels (samples/<role>.txt) so
        # images reviewed as correct can go straight into vendor fine-tuning. Never blocks grading.
        if training_labels.write_result_labels(result, sample) is not None:
            training_labels.save_class_names(product_review.REVIEW_DIR, role, model.names)
        metadata["sample_frames"].append(str(sample))
        metadata["cameras"][role] = {
            "photo": paths[role],
            "cm_per_pixel": scale,
            "model": str(MODEL_PATHS[role]),
            "runtime": runtime[1], "runtime_model": str(runtime[0]),  # TensorRT engine or the .pt
            "confidence_threshold": confidence,
            "detection_confidence": selected_confidence,
        }
        if leg_check is not None:
            metadata["cameras"][role]["leg_check"] = leg_check
    metadata["measurements"] = measurements
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(2, 2, "兩張照片推論完成")
    return measurements
