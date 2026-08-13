"""Background queue for camera recording and YOLO segmentation inference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional
from uuid import uuid4

from arduino_control import commands_for_grade, list_serial_ports, send_arduino_commands
from database import (
    add_chicken,
    get_chicken,
    get_latest_grading_config,
    update_chicken,
    update_grading_result,
)
from grading import grade_chicken, validate_grading_config
from inference_utils import (
    get_torch_cuda_status,
    record_camera_video,
    run_yolo_seg_inference_on_video,
    write_metric_to_chicken,
)


STATUS_WAITING = "等待中"
STATUS_RECORDING = "錄影中"
STATUS_INFERENCING = "推論中"
STATUS_COMPLETED = "已完成"
STATUS_FAILED = "失敗"

FINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED}
RECORDING_SECONDS = 0.5
RECORDING_TARGET_FPS = 20.0
TARGET_INFERENCE_PHOTOS = 10
PRELIMINARY_LIGHT_SECONDS = 1.0


@dataclass
class InferenceJob:
    job_id: str
    chicken_id: str
    weight_g: float
    camera_index: int
    confidence_threshold: float
    frame_stride: int
    target_photo_count: int
    cm_per_pixel: float
    hardware_enabled: bool = True
    arduino_port: str = ""
    status: str = STATUS_WAITING
    progress_completed: int = 0
    progress_total: int = 0
    message: str = "等待背景推論"
    video_id: Optional[str] = None
    video_name: Optional[str] = None
    result_row: Optional[Dict[str, object]] = None
    preliminary_grade_result: Optional[str] = None
    preliminary_light_result: Optional[str] = None
    arduino_result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


_jobs: Dict[str, InferenceJob] = {}
_queue: "Queue[str]" = Queue()
_lock = threading.Lock()
_worker_thread: Optional[threading.Thread] = None


def _touch(job: InferenceJob) -> None:
    job.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _snapshot(job: InferenceJob) -> Dict[str, object]:
    result = job.result_row or {}
    return {
        "job_id": job.job_id,
        "chicken ID": job.chicken_id,
        "\u72c0\u614b": job.status,
        "\u9032\u5ea6": f"{job.progress_completed}/{job.progress_total}" if job.progress_total else "-",
        "\u8a0a\u606f": job.message,
        "\u9ad4\u91cd(g)": result.get("weight_g", ""),
        "\u96de\u51a0\u9762\u7a4d(cm\u00b2)": result.get("comb_area_cm2", ""),
        "\u8173\u811b\u5bec\u5ea6(cm)": result.get("shank_width_cm", ""),
        "\u8173\u811b\u9577\u5ea6(cm)": result.get("shank_length_cm", ""),
        "\u521d\u6b65\u5206\u7d1a": job.preliminary_grade_result or "",
        "\u521d\u6b65\u71c8\u865f": job.preliminary_light_result or "",
        "Arduino": job.arduino_result or "",
        "\u5f71\u7247": job.video_name or "",
        "\u932f\u8aa4": job.error or "",
        "\u5efa\u7acb\u6642\u9593": job.created_at,
        "\u66f4\u65b0\u6642\u9593": job.updated_at,
    }

def list_jobs() -> List[Dict[str, object]]:
    with _lock:
        return [_snapshot(job) for job in sorted(_jobs.values(), key=lambda item: item.created_at, reverse=True)]


def get_job(job_id: str) -> Optional[InferenceJob]:
    with _lock:
        return _jobs.get(job_id)


def latest_completed_job_for_chicken(chicken_id: str) -> Optional[InferenceJob]:
    with _lock:
        matches = [
            job
            for job in _jobs.values()
            if job.chicken_id == chicken_id and job.status == STATUS_COMPLETED
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.updated_at, reverse=True)[0]


def has_active_jobs() -> bool:
    with _lock:
        return any(job.status not in FINAL_STATUSES for job in _jobs.values())


def enqueue_inference_job(
    chicken_id: str,
    weight_g: float,
    camera_index: int,
    confidence_threshold: float,
    frame_stride: int,
    cm_per_pixel: float,
    hardware_enabled: bool = True,
    arduino_port: str = "",
    target_photo_count: int = TARGET_INFERENCE_PHOTOS,
) -> str:
    job = InferenceJob(
        job_id=uuid4().hex[:12],
        chicken_id=chicken_id,
        weight_g=float(weight_g),
        camera_index=int(camera_index),
        confidence_threshold=float(confidence_threshold),
        frame_stride=int(frame_stride),
        target_photo_count=max(1, int(target_photo_count)),
        cm_per_pixel=float(cm_per_pixel),
        hardware_enabled=bool(hardware_enabled),
        arduino_port=str(arduino_port or "").strip(),
    )
    with _lock:
        _jobs[job.job_id] = job
    _queue.put(job.job_id)
    ensure_worker_running()
    return job.job_id


def ensure_worker_running() -> None:
    global _worker_thread
    with _lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_worker_loop, name="chicken-inference-worker", daemon=True)
        _worker_thread.start()


def _update_job(job_id: str, **changes: object) -> None:
    with _lock:
        job = _jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)
        _touch(job)


def _prepare_chicken_record(job: InferenceJob) -> None:
    selected = get_chicken(job.chicken_id)
    if selected:
        update_chicken(
            job.chicken_id,
            job.weight_g,
            selected["comb_area_cm2"],
            selected.get("shank_width_cm", 0.0),
            selected.get("shank_length_cm", 0.0),
        )
    else:
        add_chicken(job.chicken_id, job.weight_g, 0.0, 0.0, 0.0)


def _resolve_arduino_port(job: InferenceJob) -> str:
    if job.arduino_port:
        return job.arduino_port
    ports = list_serial_ports()
    return ports[0].device if ports else ""


def _run_preliminary_grading_and_light(job: InferenceJob, updated_row: Dict[str, object]) -> tuple[Optional[str], Optional[str], str]:
    config = get_latest_grading_config()
    if not config:
        return None, None, "\u672a\u627e\u5230\u5df2\u5132\u5b58\u5206\u7d1a\u9580\u6abb\uff0c\u7565\u904e\u521d\u6b65\u4eae\u71c8"

    selected_features, pass_thresholds, standby_thresholds = config
    validate_grading_config(selected_features, pass_thresholds, standby_thresholds)
    grade_result, light_result = grade_chicken(updated_row, selected_features, pass_thresholds, standby_thresholds)
    update_grading_result(
        job.chicken_id,
        selected_features,
        pass_thresholds,
        standby_thresholds,
        grade_result,
        light_result,
    )

    if not job.hardware_enabled:
        return grade_result, light_result, "\u5be6\u9ad4\u71c8\u865f\u672a\u555f\u7528"

    arduino_port = _resolve_arduino_port(job)
    if not arduino_port:
        return grade_result, light_result, "\u627e\u4e0d\u5230 Arduino serial port"

    commands = commands_for_grade(grade_result)
    try:
        replies = send_arduino_commands(arduino_port, commands)
        time.sleep(PRELIMINARY_LIGHT_SECONDS)
        off_replies = send_arduino_commands(arduino_port, ["LIGHT:OFF"])
    except Exception as exc:
        return grade_result, light_result, f"Arduino \u71c8\u865f\u63a7\u5236\u5931\u6557\uff1a{exc}"
    reply_text = f"\uff1b\u56de\u8986\uff1a{', '.join(replies)}" if replies else ""
    off_reply_text = f"\uff1b\u95dc\u71c8\u56de\u8986\uff1a{', '.join(off_replies)}" if off_replies else ""
    return (
        grade_result,
        light_result,
        f"\u5df2\u9001\u51fa {', '.join(commands)}\uff0c\u4eae\u71c8 {PRELIMINARY_LIGHT_SECONDS:.0f} \u79d2\u5f8c\u95dc\u9589{reply_text}{off_reply_text}",
    )

def _frame_stride_for_target_photos(video_path: Path, target_photo_count: int) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    finally:
        cap.release()
    if total_frames <= 0:
        return 1
    return max(1, total_frames // max(1, int(target_photo_count)))


def _inference_device() -> str:
    status = get_torch_cuda_status()
    return "cuda" if status.get("cuda_available") else "cpu"


def _worker_loop() -> None:
    while True:
        try:
            job_id = _queue.get(timeout=0.5)
        except Empty:
            return
        try:
            with _lock:
                job = _jobs[job_id]

            _prepare_chicken_record(job)
            _update_job(job_id, status=STATUS_RECORDING, message="正在錄製 3 秒影片")
            video_path = record_camera_video(
                chicken_id=job.chicken_id,
                camera_index=job.camera_index,
                duration_seconds=RECORDING_SECONDS,
                target_fps=RECORDING_TARGET_FPS,
            )
            frame_stride = _frame_stride_for_target_photos(video_path, job.target_photo_count)
            _update_job(
                job_id,
                video_id=video_path.stem,
                video_name=video_path.name,
                status=STATUS_INFERENCING,
                message="正在載入 YOLO segmentation 模型",
            )

            def progress_callback(completed: int, total: int, message: str) -> None:
                _update_job(
                    job_id,
                    progress_completed=int(completed),
                    progress_total=int(total),
                    message=message,
                )

            metric = run_yolo_seg_inference_on_video(
                video_path=video_path,
                progress_callback=progress_callback,
                confidence_threshold=job.confidence_threshold,
                frame_stride=frame_stride,
                device=_inference_device(),
                max_photos=job.target_photo_count,
            )
            updated_row = write_metric_to_chicken(job.chicken_id, metric, job.cm_per_pixel)
            grade_result, light_result, arduino_result = _run_preliminary_grading_and_light(job, dict(updated_row))
            final_message = (
                "推論完成，結果已寫入資料庫並完成初步篩選"
                if grade_result
                else "推論完成，結果已寫入資料庫；未執行初步亮燈"
            )
            _update_job(
                job_id,
                status=STATUS_COMPLETED,
                progress_completed=max(job.progress_completed, job.progress_total),
                message=final_message,
                result_row=dict(updated_row),
                preliminary_grade_result=grade_result,
                preliminary_light_result=light_result,
                arduino_result=arduino_result,
            )
        except Exception as exc:
            _update_job(job_id, status=STATUS_FAILED, message="任務失敗", error=str(exc))
        finally:
            _queue.task_done()
            time.sleep(0.1)
