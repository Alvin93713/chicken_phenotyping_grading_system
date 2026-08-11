"""YOLO + SAM2 影片推論與資料庫寫入整合。"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

from database import get_all_chickens, get_chicken, update_chicken


SAM_ENV_PYTHON = Path("C:/cg_sam_env/Scripts/python.exe")


def is_frozen_app() -> bool:
    """判斷目前是否在 PyInstaller exe 中執行。"""
    return bool(getattr(sys, "frozen", False))


BASE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if is_frozen_app() else BASE_DIR
BUNDLED_DATA_DIR = BASE_DIR / "data"
DATA_DIR = APP_DIR / "data" if is_frozen_app() else BUNDLED_DATA_DIR


def _asset_data_dir() -> Path:
    """Prefer user-visible assets next to the exe, then fall back to bundled assets."""
    if not is_frozen_app():
        return BUNDLED_DATA_DIR

    external_required = [
        DATA_DIR / "best.pt",
        DATA_DIR / "sam2_b.pt",
        DATA_DIR / "scripts" / "sam2_pipeline.py",
        DATA_DIR / "comb_shank_sam_video",
    ]
    if all(path.exists() for path in external_required):
        return DATA_DIR
    return BUNDLED_DATA_DIR


ASSET_DATA_DIR = _asset_data_dir()
VIDEOS_DIR = ASSET_DATA_DIR / "comb_shank_sam_video"
OUTPUTS_DIR = DATA_DIR / "sam2_inference_outputs"
YOLO_WEIGHTS = ASSET_DATA_DIR / "best.pt"
YOLO_SEG_WEIGHTS = BASE_DIR / "best.pt"
SAM2_WEIGHTS = ASSET_DATA_DIR / "sam2_b.pt"
PIPELINE_SCRIPT = ASSET_DATA_DIR / "scripts" / "sam2_pipeline.py"
SELECTED_VIDEO_INPUT_DIR = DATA_DIR / "_sam2_selected_input"
CAMERA_CAPTURE_DIR = DATA_DIR / "camera_captures"
CALIBRATION_CAPTURE_DIR = DATA_DIR / "calibration_captures"
PROGRESS_FILE = OUTPUTS_DIR / "last_inference_progress.json"
COMB_CONFIDENCE_FALLBACK = 0.25


def app_executable() -> str:
    """回傳目前 App 可用的 Python/exe 執行入口。"""
    if is_frozen_app():
        return sys.executable
    if SAM_ENV_PYTHON.exists():
        return str(SAM_ENV_PYTHON)
    return sys.executable


@dataclass
class VideoMetric:
    video_id: str
    video_name: str
    comb_area_px2_median: Optional[float]
    shank_width_px_median: Optional[float]
    comb_count: int
    shank_count: int
    shank_length_px_median: Optional[float] = None
    source: str = ""


def list_video_paths() -> List[Path]:
    """列出待推論影片，依檔名排序。"""
    if not VIDEOS_DIR.exists():
        return []
    return sorted(
        path
        for path in VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    )


def validate_inference_assets() -> List[str]:
    """檢查推論所需模型、影片與腳本是否存在。"""
    errors: List[str] = []
    required_files = {
        "YOLO 權重": YOLO_WEIGHTS,
        "SAM2 權重": SAM2_WEIGHTS,
        "推論腳本": PIPELINE_SCRIPT,
    }
    for label, path in required_files.items():
        if not path.exists():
            errors.append(f"{label}不存在：{path}")
    if not list_video_paths():
        errors.append(f"找不到影片：{VIDEOS_DIR}")
    return errors


def get_torch_cuda_status(python_executable: Optional[str] = None) -> Dict[str, object]:
    """查詢推論 Python 環境的 PyTorch / CUDA 狀態。"""
    if is_frozen_app() and (
        python_executable is None
        or Path(python_executable).resolve() == Path(sys.executable).resolve()
    ):
        try:
            import torch

            return {
                "python": sys.executable,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "python": sys.executable,
                "torch_version": None,
                "cuda_available": False,
                "cuda_device_count": 0,
                "error": str(exc),
            }

    py = python_executable or app_executable()
    command = [
        py,
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({"
            "'torch_version': torch.__version__, "
            "'cuda_available': torch.cuda.is_available(), "
            "'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0"
            "}))"
        ),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {
            "python": py,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "error": (result.stderr or result.stdout).strip(),
        }
    try:
        status = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        status = {
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "error": result.stdout.strip(),
        }
    status["python"] = py
    return status


def build_pairing_rows() -> List[Dict[str, object]]:
    """依序配對現有 chicken_id 與影片。"""
    chickens = get_all_chickens()
    videos = list_video_paths()
    rows: List[Dict[str, object]] = []
    for index, video_path in enumerate(videos):
        chicken = chickens[index] if index < len(chickens) else None
        rows.append(
            {
                "index": index + 1,
                "chicken_id": chicken["chicken_id"] if chicken else "",
                "weight_g": chicken["weight_g"] if chicken else None,
                "video_id": video_path.stem,
                "video_name": video_path.name,
            }
        )
    return rows


def get_video_for_chicken_id(chicken_id: str) -> Path:
    """依目前配對規則取得 chicken_id 對應的影片。"""
    normalized_id = str(chicken_id).strip()
    chickens = get_all_chickens()
    videos = list_video_paths()
    chicken_ids = [item["chicken_id"] for item in chickens]
    if normalized_id not in chicken_ids:
        raise ValueError(f"資料庫查無 chicken_id，無法配對影片：{normalized_id}")

    index = chicken_ids.index(normalized_id)
    if index >= len(videos):
        raise RuntimeError(
            f"chicken_id {normalized_id} 位於第 {index + 1} 筆，但目前只有 {len(videos)} 支影片可配對。"
        )
    return videos[index]


def prepare_single_video_dir(video_path: Path) -> Path:
    """建立只包含指定影片的暫存資料夾，供 pipeline 單影片推論。"""
    SELECTED_VIDEO_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in SELECTED_VIDEO_INPUT_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()

    target_path = SELECTED_VIDEO_INPUT_DIR / video_path.name
    shutil.copy2(video_path, target_path)
    return SELECTED_VIDEO_INPUT_DIR


def get_video_processed_frame_count(video_path: Path, frame_stride: int) -> int:
    """Return how many frames will be sent through YOLO + SAM2 for this video."""
    import cv2

    stride = max(1, int(frame_stride))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片以計算推論照片數：{video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    finally:
        cap.release()
    if total_frames <= 0:
        return 0
    return ((total_frames - 1) // stride) + 1


def validate_yolo_seg_assets() -> List[str]:
    """Validate assets needed by the camera YOLO segmentation workflow."""
    errors: List[str] = []
    if not YOLO_SEG_WEIGHTS.exists():
        errors.append(f"找不到 YOLO segmentation 權重：{YOLO_SEG_WEIGHTS}")
    return errors


def record_camera_video(
    chicken_id: str,
    camera_index: int = 0,
    duration_seconds: float = 3.0,
    target_fps: float = 20.0,
) -> Path:
    """Record a short video from a local camera and return the saved path."""
    import cv2

    CAMERA_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_chicken_id = "".join(ch for ch in str(chicken_id) if ch.isalnum() or ch in {"-", "_"})
    video_path = CAMERA_CAPTURE_DIR / f"{safe_chicken_id}_{timestamp}.mp4"

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機 ID {camera_index}。請確認外接鏡頭已連接，或調整 camera ID。")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("攝影機已開啟，但無法讀取畫面。")

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(target_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"無法建立錄影檔：{video_path}")

        frame_interval = 1.0 / max(1.0, float(target_fps))
        start_time = time.time()
        next_frame_time = start_time
        frames_written = 0
        try:
            while time.time() - start_time < float(duration_seconds):
                if frames_written == 0:
                    current_frame = frame
                else:
                    ok, current_frame = cap.read()
                    if not ok or current_frame is None:
                        continue
                writer.write(current_frame)
                frames_written += 1

                next_frame_time += frame_interval
                sleep_seconds = next_frame_time - time.time()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        finally:
            writer.release()

        if frames_written == 0:
            raise RuntimeError("錄影失敗：未寫入任何影格。")
    finally:
        cap.release()

    return video_path


def capture_calibration_image(camera_index: int = 0) -> Path:
    """Capture one still image from the local camera for scale calibration."""
    import cv2

    CALIBRATION_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = CALIBRATION_CAPTURE_DIR / f"calibration_{timestamp}.png"

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機 ID {camera_index}。請確認鏡頭已連接，或調整 camera ID。")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("攝影機已開啟，但無法拍攝校正照片。")
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("無法將校正照片編碼為 PNG。")
        encoded.tofile(str(image_path))
        if not image_path.exists() or image_path.stat().st_size == 0:
            raise RuntimeError(f"無法儲存校正照片：{image_path}")
    finally:
        cap.release()

    return image_path


def capture_camera_frame(camera_index: int = 0):
    """Capture one BGR frame from a local camera."""
    import cv2

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機 ID {camera_index}。")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("攝影機已開啟，但無法讀取畫面。")
        return frame
    finally:
        cap.release()


@lru_cache(maxsize=1)
def get_yolo_seg_model():
    """Load the YOLO segmentation model once per Python process."""
    from ultralytics import YOLO

    errors = validate_yolo_seg_assets()
    if errors:
        raise FileNotFoundError("\n".join(errors))
    return YOLO(str(YOLO_SEG_WEIGHTS))


def annotate_yolo_seg_frame(frame_bgr, confidence_threshold: float = 0.5, device: str = "cpu"):
    """Run YOLO segmentation on one BGR frame and return annotated RGB frame."""
    import cv2

    model = get_yolo_seg_model()
    result = model.predict(
        source=frame_bgr,
        conf=float(confidence_threshold),
        device=device,
        verbose=False,
    )[0]
    annotated_bgr = result.plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def _class_name_is(name: object, expected: str) -> bool:
    return str(name).strip().lower() == expected.lower()


def _mask_sides_px(mask) -> tuple[Optional[float], Optional[float]]:
    """Compute short and long sides of a mask using its minimum-area rectangle."""
    import cv2
    import numpy as np

    mask_bool = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None, None
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if len(points) < 3:
        return 1.0, 1.0
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    short_side = min(float(width), float(height))
    long_side = max(float(width), float(height))
    return (
        short_side if short_side > 0 else None,
        long_side if long_side > 0 else None,
    )


def run_yolo_seg_inference_on_video(
    video_path: Path,
    progress_callback: Callable[[int, int, str], None],
    confidence_threshold: float = 0.25,
    frame_stride: int = 1,
    device: str = "cpu",
    max_photos: Optional[int] = None,
) -> VideoMetric:
    """Run the local YOLO segmentation model on a recorded camera video."""
    import cv2
    import numpy as np

    errors = validate_yolo_seg_assets()
    if errors:
        raise FileNotFoundError("\n".join(errors))

    stride = max(1, int(frame_stride))
    video_id = video_path.stem
    video_out_dir = OUTPUTS_DIR / video_id
    samples_dir = video_out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in samples_dir.glob("*.png"):
        old_sample.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟錄影檔：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total_photos = ((total_frames - 1) // stride + 1) if total_frames > 0 else 0
    if max_photos is not None:
        total_photos = min(total_photos, max(1, int(max_photos)))
    progress_callback(0, total_photos, "正在載入 YOLO segmentation 模型...")

    model = get_yolo_seg_model()
    names = getattr(model, "names", {}) or {}
    shank_confidence_threshold = float(confidence_threshold)
    comb_confidence_threshold = min(float(confidence_threshold), COMB_CONFIDENCE_FALLBACK)
    model_confidence_threshold = min(shank_confidence_threshold, comb_confidence_threshold)

    metadata: Dict[str, object] = {
        "video": video_path.name,
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0,
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0,
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0,
        "total_frames": total_frames,
        "processed_frames": 0,
        "frame_stride": stride,
        "detections_per_class": {"Comb": 0, "Shank": 0},
        "annotated_video": "",
        "sample_frames": [],
        "frames": [],
        "source": "camera_yolo_seg",
        "weights": str(YOLO_SEG_WEIGHTS),
        "model_confidence_threshold": model_confidence_threshold,
        "comb_confidence_threshold": comb_confidence_threshold,
        "shank_confidence_threshold": shank_confidence_threshold,
    }

    comb_areas: List[float] = []
    shank_widths: List[float] = []
    shank_lengths: List[float] = []
    frame_index = 0
    completed = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue
            if max_photos is not None and completed >= int(max_photos):
                break

            result = model.predict(
                source=frame_bgr,
                conf=model_confidence_threshold,
                device=device,
                verbose=False,
            )[0]
            annotated = result.plot()
            sample_path = samples_dir / f"frame_{frame_index:05d}.png"
            cv2.imwrite(str(sample_path), annotated)
            metadata["sample_frames"].append(str(sample_path))  # type: ignore[union-attr]

            frame_record: Dict[str, object] = {"frame_index": frame_index, "detections": []}
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            shank_candidates: List[Dict[str, object]] = []
            if boxes is not None and masks is not None and masks.data is not None:
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                bboxes = boxes.xyxy.detach().cpu().numpy()
                mask_stack = masks.data.detach().cpu().numpy()

                for cls_id, confidence, bbox, mask in zip(class_ids, confidences, bboxes, mask_stack):
                    raw_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    if _class_name_is(raw_name, "comb"):
                        class_name = "Comb"
                    elif _class_name_is(raw_name, "shank"):
                        class_name = "Shank"
                    else:
                        continue
                    if class_name == "Comb" and float(confidence) < comb_confidence_threshold:
                        continue
                    if class_name == "Shank" and float(confidence) < shank_confidence_threshold:
                        continue

                    mask_area = int(np.asarray(mask).astype(bool).sum())
                    shank_width_px = None
                    shank_length_px = None
                    if class_name == "Comb":
                        comb_areas.append(float(mask_area))
                    elif class_name == "Shank":
                        shank_width_px, shank_length_px = _mask_sides_px(mask)

                    detections_per_class = metadata["detections_per_class"]
                    detections_per_class[class_name] = int(detections_per_class[class_name]) + 1  # type: ignore[index]
                    det_record = {
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "confidence": round(float(confidence), 4),
                        "bbox_xyxy": [round(float(value), 2) for value in bbox],
                        "bbox_width_px": round(float(bbox[2] - bbox[0]), 2),
                        "bbox_height_px": round(float(bbox[3] - bbox[1]), 2),
                        "mask_area": mask_area,
                    }
                    if shank_width_px is not None:
                        det_record["shank_width_px"] = round(float(shank_width_px), 2)
                    if shank_length_px is not None:
                        det_record["shank_length_px"] = round(float(shank_length_px), 2)
                    frame_record["detections"].append(det_record)  # type: ignore[union-attr]
                    if class_name == "Shank" and shank_width_px is not None:
                        shank_candidates.append(
                            {
                                "mask_area": mask_area,
                                "shank_width_px": float(shank_width_px),
                                "shank_length_px": float(shank_length_px) if shank_length_px is not None else None,
                            }
                        )

            if shank_candidates:
                largest_shank = max(shank_candidates, key=lambda item: int(item["mask_area"]))
                shank_widths.append(float(largest_shank["shank_width_px"]))
                if largest_shank.get("shank_length_px") is not None:
                    shank_lengths.append(float(largest_shank["shank_length_px"]))

            metadata["frames"].append(frame_record)  # type: ignore[union-attr]
            completed += 1
            metadata["processed_frames"] = completed
            progress_callback(completed, total_photos, f"已完成 {completed} / {total_photos} 張照片")
            frame_index += 1
    finally:
        cap.release()

    metadata_path = video_out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    progress_callback(completed, total_photos, f"推論程序結束：已完成 {completed} / {total_photos} 張照片")
    return VideoMetric(
        video_id=video_id,
        video_name=video_path.name,
        comb_area_px2_median=_median(comb_areas),
        shank_width_px_median=_median(shank_widths),
        comb_count=len(comb_areas),
        shank_count=len(shank_widths),
        shank_length_px_median=_median(shank_lengths),
        source="YOLO-seg camera",
    )


def run_yolo_sam2_inference(
    python_executable: Optional[str] = None,
    confidence_threshold: float = 0.25,
    sample_frames: int = 3,
    frame_stride: int = 5,
    device: str = "cpu",
    skip_existing: bool = True,
    skip_annotated_video: bool = True,
    videos_dir: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    """呼叫既有 sam2_pipeline.py 執行 YOLO bbox + SAM2 mask 推論。"""
    command = build_yolo_sam2_command(
        python_executable=python_executable,
        confidence_threshold=confidence_threshold,
        sample_frames=sample_frames,
        frame_stride=frame_stride,
        device=device,
        skip_existing=skip_existing,
        skip_annotated_video=skip_annotated_video,
        videos_dir=videos_dir,
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        command,
        cwd=str(BASE_DIR),
        text=True,
        capture_output=True,
        check=False,
    )


def build_yolo_sam2_command(
    python_executable: Optional[str] = None,
    confidence_threshold: float = 0.25,
    sample_frames: int = 3,
    frame_stride: int = 5,
    device: str = "cpu",
    skip_existing: bool = True,
    skip_annotated_video: bool = True,
    videos_dir: Optional[Path] = None,
) -> List[str]:
    """建立 YOLO + SAM2 推論命令。"""
    errors = validate_inference_assets()
    if errors:
        raise FileNotFoundError("\n".join(errors))

    py = python_executable or app_executable()
    cuda_status = get_torch_cuda_status(py)
    if device == "cuda" and not cuda_status.get("cuda_available"):
        device = "cpu"

    command = [py]
    if is_frozen_app() and python_executable is None:
        command.append("--run-sam2-pipeline")
    else:
        command.append(str(PIPELINE_SCRIPT))

    command.extend(
        [
        "run-inference",
        "--videos-dir",
        str(videos_dir or VIDEOS_DIR),
        "--output-dir",
        str(OUTPUTS_DIR),
        "--yolo-weights",
        str(YOLO_WEIGHTS),
        "--sam-weights",
        str(SAM2_WEIGHTS),
        "--confidence-threshold",
        str(confidence_threshold),
        "--sample-frames",
        str(sample_frames),
        "--frame-stride",
        str(max(1, int(frame_stride))),
        "--device",
        device,
        "--progress-file",
        str(PROGRESS_FILE),
        ]
    )
    if skip_existing:
        command.append("--skip-existing")
    if skip_annotated_video:
        command.append("--skip-annotated-video")
    return command


def _completed_metadata_count(start_time: float) -> int:
    count = 0
    for metadata_path in OUTPUTS_DIR.glob("*/metadata.json"):
        try:
            if metadata_path.stat().st_mtime >= start_time:
                count += 1
        except OSError:
            continue
    return count


def _read_progress_frames() -> int:
    if not PROGRESS_FILE.exists():
        return 0
    try:
        with PROGRESS_FILE.open("r", encoding="utf-8") as file:
            progress = json.load(file)
        return int(progress.get("completed_frames") or 0)
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return 0


def run_yolo_sam2_inference_with_progress(
    progress_callback: Callable[[int, int, str], None],
    python_executable: Optional[str] = None,
    confidence_threshold: float = 0.25,
    sample_frames: int = 3,
    frame_stride: int = 5,
    device: str = "cpu",
    skip_existing: bool = False,
    skip_annotated_video: bool = True,
    videos_dir: Optional[Path] = None,
    total_videos: Optional[int] = None,
    total_photos: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    """執行推論並以完成推論照片數回報進度。"""
    command = build_yolo_sam2_command(
        python_executable=python_executable,
        confidence_threshold=confidence_threshold,
        sample_frames=sample_frames,
        frame_stride=frame_stride,
        device=device,
        skip_existing=skip_existing,
        skip_annotated_video=skip_annotated_video,
        videos_dir=videos_dir,
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    total_units = total_photos if total_photos is not None else total_videos
    total_units = total_units if total_units is not None else len(list_video_paths())
    start_time = time.time()
    progress_callback(0, total_units, "正在載入 YOLO 與 SAM2 模型...")

    stdout_path = OUTPUTS_DIR / "last_inference_stdout.log"
    stderr_path = OUTPUTS_DIR / "last_inference_stderr.log"
    stdout_file = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_file = stderr_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        text=True,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    last_completed = -1

    try:
        while process.poll() is None:
            completed = _read_progress_frames()
            if completed != last_completed:
                last_completed = completed
                progress_callback(completed, total_units, f"已完成 {completed} / {total_units} 張照片")
            time.sleep(1)
    finally:
        stdout_file.close()
        stderr_file.close()

    completed = max(_read_progress_frames(), _completed_metadata_count(start_time))
    if total_photos is not None and process.returncode == 0:
        completed = max(completed, total_photos)
    progress_callback(completed, total_units, f"推論程序結束：已完成 {completed} / {total_units} 張照片")
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def read_video_metrics(outputs_dir: Path = OUTPUTS_DIR) -> List[VideoMetric]:
    """從 metadata.json 擷取 Comb/Shank mask area 中位數。"""
    metadata_paths = sorted(outputs_dir.glob("*/metadata.json"))
    metrics: List[VideoMetric] = []

    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        video_name = str(metadata.get("video") or f"{metadata_path.parent.name}.mp4")
        comb_areas: List[float] = []
        shank_widths: List[float] = []
        shank_lengths: List[float] = []

        for frame in metadata.get("frames", []):
            for detection in frame.get("detections", []):
                if detection.get("class_name") == "Comb":
                    area = detection.get("mask_area")
                    if area is None:
                        continue
                    comb_areas.append(float(area))
                elif detection.get("class_name") == "Shank":
                    width = detection.get("shank_width_px")
                    if width is not None:
                        shank_widths.append(float(width))
                    length = detection.get("shank_length_px")
                    if length is not None:
                        shank_lengths.append(float(length))

        metrics.append(
            VideoMetric(
                video_id=Path(video_name).stem,
                video_name=video_name,
                comb_area_px2_median=_median(comb_areas),
                shank_width_px_median=_median(shank_widths),
                comb_count=len(comb_areas),
                shank_count=len(shank_widths),
                shank_length_px_median=_median(shank_lengths),
                source="YOLO+SAM2",
            )
        )

    return sorted(metrics, key=lambda item: item.video_name)


def metrics_to_rows(metrics: List[VideoMetric], cm_per_pixel: float) -> List[Dict[str, object]]:
    """轉成 Streamlit 顯示用 rows。"""
    rows: List[Dict[str, object]] = []
    area_factor = cm_per_pixel**2
    for metric in metrics:
        comb_cm2 = (
            metric.comb_area_px2_median * area_factor
            if metric.comb_area_px2_median is not None
            else None
        )
        shank_width_cm = (
            metric.shank_width_px_median * cm_per_pixel
            if metric.shank_width_px_median is not None
            else None
        )
        shank_length_cm = (
            metric.shank_length_px_median * cm_per_pixel
            if metric.shank_length_px_median is not None
            else None
        )
        row = {
            "video_id": metric.video_id,
            "video_name": metric.video_name,
            "comb_area_cm2": comb_cm2,
            "shank_width_cm": shank_width_cm,
            "shank_length_cm": shank_length_cm,
            "comb_count": metric.comb_count,
            "shank_count": metric.shank_count,
            "source": metric.source,
        }
        rows.append(row)
    return rows


def read_video_metric(video_id: str, outputs_dir: Path = OUTPUTS_DIR) -> VideoMetric:
    """Read one video's YOLO + SAM2 metric from the output metadata."""
    normalized_id = Path(str(video_id).strip()).stem
    for metric in read_video_metrics(outputs_dir):
        if metric.video_id == normalized_id or Path(metric.video_name).stem == normalized_id:
            return metric
    raise RuntimeError(f"找不到影片 {normalized_id} 的推論 metadata，請確認推論是否完成。")


def write_metric_to_chicken(
    chicken_id: str,
    metric: VideoMetric,
    cm_per_pixel: float,
) -> Dict[str, object]:
    """Write one video's inferred measurements back to one chicken record."""
    if cm_per_pixel <= 0:
        raise ValueError("cm_per_pixel must be greater than 0")

    chicken = get_chicken(chicken_id)
    if not chicken:
        raise RuntimeError(f"找不到 chicken_id：{chicken_id}")

    area_factor = cm_per_pixel**2
    comb_area_cm2 = (
        round(metric.comb_area_px2_median * area_factor, 4)
        if metric.comb_area_px2_median is not None
        else None
    )
    shank_width_cm = (
        round(metric.shank_width_px_median * cm_per_pixel, 4)
        if metric.shank_width_px_median is not None
        else None
    )
    shank_length_cm = (
        round(metric.shank_length_px_median * cm_per_pixel, 4)
        if metric.shank_length_px_median is not None
        else None
    )
    missing_metrics = []
    if comb_area_cm2 is None:
        missing_metrics.append("comb_area_cm2")
    if shank_width_cm is None:
        missing_metrics.append("shank_width_cm")
    if shank_length_cm is None:
        missing_metrics.append("shank_length_cm")

    update_chicken(
        chicken_id,
        chicken["weight_g"],
        comb_area_cm2,
        shank_width_cm,
        shank_length_cm,
    )
    return {
        "chicken ID": chicken_id,
        "video_name": metric.video_name,
        "weight_g": chicken["weight_g"],
        "comb_area_cm2": comb_area_cm2,
        "shank_width_cm": shank_width_cm,
        "shank_length_cm": shank_length_cm,
        "missing_metrics": ", ".join(missing_metrics),
    }

def write_metrics_to_database(cm_per_pixel: float) -> List[Dict[str, object]]:
    """依序將影片中位數面積寫入現有 chickens 資料表，體重保持不變。"""
    if cm_per_pixel <= 0:
        raise ValueError("cm_per_pixel 必須大於 0。")

    chickens = get_all_chickens()
    metrics = read_video_metrics()
    if not metrics:
        raise RuntimeError("找不到推論 metadata，請先執行 YOLO + SAM2 推論。")
    if len(metrics) < len(chickens):
        raise RuntimeError(f"推論結果只有 {len(metrics)} 筆，少於資料庫雞隻數 {len(chickens)}。")

    area_factor = cm_per_pixel**2
    updated: List[Dict[str, object]] = []
    for chicken, metric in zip(chickens, metrics):
        comb_area_cm2 = (
            round(metric.comb_area_px2_median * area_factor, 4)
            if metric.comb_area_px2_median is not None
            else None
        )
        shank_width_cm = (
            round(metric.shank_width_px_median * cm_per_pixel, 4)
            if metric.shank_width_px_median is not None
            else None
        )
        shank_length_cm = (
            round(metric.shank_length_px_median * cm_per_pixel, 4)
            if metric.shank_length_px_median is not None
            else None
        )
        missing_metrics = []
        if comb_area_cm2 is None:
            missing_metrics.append("comb_area_cm2")
        if shank_width_cm is None:
            missing_metrics.append("shank_width_cm")
        if shank_length_cm is None:
            missing_metrics.append("shank_length_cm")
        update_chicken(
            chicken["chicken_id"],
            chicken["weight_g"],
            comb_area_cm2,
            shank_width_cm,
            shank_length_cm,
        )
        updated.append(
            {
                "chicken_id": chicken["chicken_id"],
                "video_name": metric.video_name,
                "weight_g": chicken["weight_g"],
                "comb_area_cm2": comb_area_cm2,
                "shank_width_cm": shank_width_cm,
                "shank_length_cm": shank_length_cm,
                "comb_area_px2_median": metric.comb_area_px2_median,
                "shank_width_px_median": metric.shank_width_px_median,
                "shank_length_px_median": metric.shank_length_px_median,
                "missing_metrics": ", ".join(missing_metrics),
            }
        )
    return updated

