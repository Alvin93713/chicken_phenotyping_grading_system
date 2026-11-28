"""Background queue for dual-camera snapshot YOLO segmentation inference (one photo per camera)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional
from uuid import uuid4

from arduino_control import commands_for_grade, light_for, list_serial_ports
from database import add_chicken, get_chicken, get_current_breed, update_chicken
from grading import grade_and_store_with_latest_config
from dual_camera import infer_pair
from inference_utils import OUTPUTS_DIR
from product_review import record_snapshot_review


STATUS_WAITING = "等待中"
STATUS_INFERENCING = "推論中"
STATUS_COMPLETED = "已完成"
STATUS_FAILED = "失敗"

FINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED}
PRELIMINARY_LIGHT_SECONDS = 1.0


@dataclass
class InferenceJob:
    job_id: str
    chicken_id: str
    weight_g: float
    confidence_threshold: float
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
    snapshot_paths: Optional[Dict[str, str]] = None
    snapshot_scales: Optional[Dict[str, float]] = None
    breed: Optional[str] = None  # the current breed when the photos were taken (a later switch does not move it)


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
        return max(matches, key=lambda item: item.updated_at, default=None)


def has_active_jobs() -> bool:
    with _lock:
        return any(job.status not in FINAL_STATUSES for job in _jobs.values())


def ensure_worker_running() -> None:
    global _worker_thread
    with _lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_worker_loop, name="chicken-inference-worker", daemon=True)
        _worker_thread.start()


def enqueue_snapshot_job(chicken_id, weight_g, paths, scales, confidence, hardware_enabled, arduino_port, breed=None):
    job = InferenceJob(
        breed=breed or get_current_breed(),
        job_id=uuid4().hex[:12], chicken_id=chicken_id, weight_g=float(weight_g),
        confidence_threshold=float(confidence),
        hardware_enabled=hardware_enabled, arduino_port=arduino_port,
        snapshot_paths=dict(paths), snapshot_scales=dict(scales),
        video_id=Path(paths["comb"]).parent.name,
        video_name=Path(paths["comb"]).parent.name + ".png", progress_total=2,
    )
    with _lock:
        _jobs[job.job_id] = job
    _queue.put(job.job_id)
    ensure_worker_running()
    return job.job_id


def _process_snapshot(job):
    _update_job(job.job_id, status=STATUS_INFERENCING, message="正在推論雙相機照片")
    measurements = infer_pair(
        job.snapshot_paths, job.snapshot_scales, job.confidence_threshold,
        lambda done, total, message: _update_job(job.job_id, progress_completed=done,
                                                progress_total=total, message=message),
    )
    existing = get_chicken(job.chicken_id)
    write = update_chicken if existing else add_chicken
    breed = (existing or {}).get("breed") or job.breed  # an existing chicken keeps its breed
    write(job.chicken_id, job.weight_g, measurements["comb_area_cm2"],
          measurements["shank_width_cm"], measurements["shank_length_cm"], breed=breed)
    row = {"chicken_id": job.chicken_id, "breed": breed, "weight_g": job.weight_g, **measurements}
    grade, light, hardware = _run_preliminary_grading_and_light(job, row)
    _update_job(job.job_id, status=STATUS_COMPLETED, progress_completed=2,
                message="雙相機推論完成，結果已寫入資料庫", result_row=row,
                preliminary_grade_result=grade, preliminary_light_result=light, arduino_result=hardware)
    _capture_product_review(job, measurements=measurements)


def _capture_product_review(job, measurements=None, error: str = "") -> None:
    """Capture optional QA data without changing the grading outcome."""
    if not job.snapshot_paths:
        return
    try:
        output_folder = OUTPUTS_DIR / Path(job.snapshot_paths["comb"]).parent.name
        record_snapshot_review(job, output_folder, measurements=measurements, error=error)
    except Exception as review_exc:
        print(f"Product review capture failed: {review_exc}")


def _update_job(job_id: str, **changes: object) -> None:
    with _lock:
        job = _jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)
        _touch(job)


def _resolve_arduino_port(job: InferenceJob) -> str:
    if job.arduino_port:
        return job.arduino_port
    ports = list_serial_ports()
    return ports[0].device if ports else ""


def _run_preliminary_grading_and_light(job: InferenceJob, updated_row: Dict[str, object]) -> tuple[Optional[str], Optional[str], str]:
    grade_result, light_result = grade_and_store_with_latest_config(updated_row)
    if grade_result is None:
        return None, None, "\u672a\u627e\u5230\u5df2\u5132\u5b58\u5206\u7d1a\u9580\u6abb\uff0c\u7565\u904e\u521d\u6b65\u4eae\u71c8"

    if not job.hardware_enabled:
        return grade_result, light_result, "\u5be6\u9ad4\u71c8\u865f\u672a\u555f\u7528"

    arduino_port = _resolve_arduino_port(job)
    if not arduino_port:
        return grade_result, light_result, "\u627e\u4e0d\u5230 Arduino serial port"

    commands = commands_for_grade(grade_result)
    try:
        replies = light_for(arduino_port, commands, PRELIMINARY_LIGHT_SECONDS)
    except Exception as exc:
        return grade_result, light_result, f"Arduino \u71c8\u865f\u63a7\u5236\u5931\u6557\uff1a{exc}"
    reply_text = f"\uff1b\u56de\u8986\uff1a{', '.join(replies)}" if replies else ""
    return (
        grade_result,
        light_result,
        f"\u5df2\u9001\u51fa {', '.join(commands)}\uff0c\u4eae\u71c8 {PRELIMINARY_LIGHT_SECONDS:.0f} \u79d2\u5f8c\u95dc\u9589{reply_text}",
    )


def _worker_loop() -> None:
    while True:
        job = None
        try:
            job_id = _queue.get(timeout=0.5)
        except Empty:
            continue
        try:
            with _lock:
                job = _jobs[job_id]
            _process_snapshot(job)
        except Exception as exc:
            _update_job(job_id, status=STATUS_FAILED, message="任務失敗", error=str(exc))
            if job is not None:
                _capture_product_review(job, error=str(exc))
        finally:
            _queue.task_done()
            time.sleep(0.1)
