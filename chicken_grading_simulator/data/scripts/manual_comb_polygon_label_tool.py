#!/usr/bin/env python3
"""Browser tool for manual Comb polygon labels on one video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, urlparse

import cv2


ROOT = Path("/home/nas2/Workspace/Hank")
DEFAULT_VIDEO_ID = "video_20250908_083233"
DEFAULT_START_FRAME = 18
DEFAULT_END_FRAME = 126
CM_PER_PX_SAM2 = 0.02
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "SAM2/results/outputs_best_videos/analysis/comb_size_timeseries/manual_labels"
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comb Polygon Label Tool</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c0c0c;
      --panel: #181818;
      --panel-2: #242424;
      --line: #3a3a3a;
      --text: #f2f2f2;
      --muted: #b7b7b7;
      --red: #ff0000;
      --blue: #18a7e0;
      --warn: #f0a236;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      width: 100vw;
      height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .toolbar {
      height: 58px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    .toolbar .spacer { flex: 1; }
    button, input, label.toggle {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
      font-size: 14px;
    }
    button {
      min-width: 42px;
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover, label.toggle:hover { background: #303030; }
    button.primary { border-color: #5d5d5d; background: #2c2c2c; }
    button.danger { border-color: #6b2b2b; color: #ffdede; }
    input[type="number"] {
      width: 82px;
      padding: 0 8px;
    }
    label.toggle {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 0 10px;
    }
    label.toggle input { margin: 0; }
    .stat {
      color: var(--muted);
      font-size: 13px;
      display: inline-flex;
      gap: 4px;
      align-items: baseline;
    }
    .stat strong {
      color: var(--text);
      font-weight: 650;
    }
    .stage {
      position: relative;
      width: 100vw;
      height: calc(100vh - 58px);
      overflow: hidden;
      background: #060606;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }
    .footer {
      position: absolute;
      left: 12px;
      right: 12px;
      bottom: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
    }
    .badge {
      max-width: 48vw;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(18, 18, 18, 0.88);
      color: var(--muted);
      font-size: 13px;
    }
    .ok { color: var(--blue); }
    .warn { color: var(--warn); }
    .empty { color: #ff7676; }
    @media (max-width: 920px) {
      .toolbar { overflow-x: auto; }
      .badge { max-width: 70vw; }
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="prev">Prev</button>
    <input id="jump" type="number" min="0" step="1">
    <button id="go">Go</button>
    <button id="next">Next</button>
    <span class="stat">Frame <strong id="frame">-</strong></span>
    <span class="stat">Done <strong id="done">0/0</strong></span>
    <span class="stat">Points <strong id="points">0</strong></span>
    <span class="stat">Area <strong id="area">-</strong></span>
    <span class="spacer"></span>
    <button id="undo">Undo</button>
    <button id="clear" class="danger">Clear</button>
    <button id="noComb">No Comb</button>
    <button id="fit">Fit</button>
    <button id="zoomIn">+</button>
    <button id="zoomOut">-</button>
    <button id="save" class="primary">Save</button>
    <label class="toggle"><input type="checkbox" id="autoNext" checked> Auto next</label>
  </div>
  <div class="stage">
    <canvas id="canvas"></canvas>
    <div class="footer">
      <div class="badge" id="status">Loading</div>
      <div class="badge" id="path"></div>
    </div>
  </div>
  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const frameEl = document.getElementById('frame');
    const doneEl = document.getElementById('done');
    const pointsEl = document.getElementById('points');
    const areaEl = document.getElementById('area');
    const statusEl = document.getElementById('status');
    const pathEl = document.getElementById('path');
    const jumpEl = document.getElementById('jump');
    const autoNextEl = document.getElementById('autoNext');

    let state = null;
    let current = null;
    let image = new Image();
    let imageReady = false;
    let draft = [];
    let zoom = 1;
    let fitZoom = 1;
    let panX = 0;
    let panY = 0;
    let lastMouse = null;
    let dragging = false;
    let dragStart = null;
    let dragMoved = false;

    function dpr() {
      return window.devicePixelRatio || 1;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr()));
      canvas.height = Math.max(1, Math.round(rect.height * dpr()));
      draw();
    }

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * dpr(),
        y: (event.clientY - rect.top) * dpr()
      };
    }

    function labelFor(frame) {
      return state.labels[String(frame)] || null;
    }

    function clonePolygon(poly) {
      return (poly || []).map((p) => [Number(p[0]), Number(p[1])]);
    }

    function polygonArea(poly) {
      if (poly.length < 3) return 0;
      let sum = 0;
      for (let i = 0; i < poly.length; i++) {
        const [x1, y1] = poly[i];
        const [x2, y2] = poly[(i + 1) % poly.length];
        sum += x1 * y2 - x2 * y1;
      }
      return Math.abs(sum) * 0.5;
    }

    function screenToImage(sx, sy) {
      return [(sx - panX) / zoom, (sy - panY) / zoom];
    }

    function imageToScreen(x, y) {
      return [panX + x * zoom, panY + y * zoom];
    }

    function setStatus(text, cls) {
      statusEl.textContent = text;
      statusEl.className = `badge ${cls || ''}`;
    }

    function frameList() {
      if (state && Array.isArray(state.frames) && state.frames.length) return state.frames;
      const frames = [];
      for (let f = state.start_frame; f <= state.end_frame; f++) frames.push(f);
      return frames;
    }

    function frameIndex(frame) {
      const frames = frameList();
      const exact = frames.indexOf(Number(frame));
      if (exact >= 0) return exact;
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < frames.length; i++) {
        const dist = Math.abs(frames[i] - Number(frame));
        if (dist < bestDist) {
          best = i;
          bestDist = dist;
        }
      }
      return best;
    }

    function selectedFrame(frame, offset) {
      const frames = frameList();
      const idx = Math.max(0, Math.min(frames.length - 1, frameIndex(frame) + offset));
      return frames[idx];
    }

    function firstUnfinishedFrame() {
      const frames = frameList();
      for (const f of frames) {
        const row = labelFor(f);
        if (!row || !row.status) return f;
      }
      return frames[0];
    }

    function updateStats() {
      if (!state) return;
      const frames = frameList();
      const total = frames.length;
      let labeled = 0;
      let noComb = 0;
      for (const f of frames) {
        const row = labelFor(f);
        if (row && row.status === 'labeled') labeled++;
        if (row && row.status === 'no_comb') noComb++;
      }
      const label = labelFor(current);
      frameEl.textContent = `${current} (${frameIndex(current) + 1}/${total})`;
      doneEl.textContent = `${labeled}+${noComb}/${total}`;
      pointsEl.textContent = String(draft.length);
      const area = polygonArea(draft);
      areaEl.textContent = area > 0 ? `${area.toFixed(1)} px2` : '-';
      jumpEl.min = String(state.start_frame);
      jumpEl.max = String(state.end_frame);
      jumpEl.value = String(current);
      if (label && label.status === 'labeled') setStatus('labeled', 'ok');
      else if (label && label.status === 'no_comb') setStatus('no comb', 'warn');
      else setStatus('empty', 'empty');
      pathEl.textContent = state.labels_json;
    }

    function fit() {
      if (!imageReady) return;
      fitZoom = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
      zoom = fitZoom;
      panX = (canvas.width - image.naturalWidth * zoom) / 2;
      panY = (canvas.height - image.naturalHeight * zoom) / 2;
      draw();
    }

    function zoomAt(multiplier, sx, sy) {
      if (!imageReady) return;
      const [ix, iy] = screenToImage(sx, sy);
      zoom = Math.max(fitZoom * 0.55, Math.min(zoom * multiplier, fitZoom * 16));
      panX = sx - ix * zoom;
      panY = sy - iy * zoom;
      draw();
    }

    function drawPolygon(poly) {
      if (!poly.length) return;
      ctx.save();
      ctx.lineWidth = Math.max(2, Math.min(5, 2.5 * Math.sqrt(zoom / Math.max(fitZoom, 0.0001))));
      ctx.strokeStyle = '#ff0000';
      ctx.fillStyle = 'rgba(255, 0, 0, 0.32)';
      ctx.beginPath();
      for (let i = 0; i < poly.length; i++) {
        const [sx, sy] = imageToScreen(poly[i][0], poly[i][1]);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      if (poly.length >= 3) ctx.closePath();
      ctx.fill();
      ctx.stroke();
      for (let i = 0; i < poly.length; i++) {
        const [sx, sy] = imageToScreen(poly[i][0], poly[i][1]);
        ctx.fillStyle = i === 0 ? '#18a7e0' : '#ff0000';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawCrosshair() {
      if (!lastMouse || !imageReady) return;
      const [ix, iy] = screenToImage(lastMouse.x, lastMouse.y);
      if (ix < 0 || iy < 0 || ix > image.naturalWidth || iy > image.naturalHeight) return;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.42)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, lastMouse.y);
      ctx.lineTo(canvas.width, lastMouse.y);
      ctx.moveTo(lastMouse.x, 0);
      ctx.lineTo(lastMouse.x, canvas.height);
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#060606';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!imageReady) return;
      ctx.imageSmoothingEnabled = zoom < 2.5;
      ctx.drawImage(image, panX, panY, image.naturalWidth * zoom, image.naturalHeight * zoom);
      drawPolygon(draft);
      drawCrosshair();
    }

    async function loadState() {
      const res = await fetch('state');
      state = await res.json();
      current = firstUnfinishedFrame();
      await loadFrame(current);
    }

    async function refreshState() {
      const res = await fetch('state');
      state = await res.json();
      updateStats();
    }

    async function loadFrame(frame) {
      current = selectedFrame(frame, 0);
      imageReady = false;
      const row = labelFor(current);
      draft = row && row.status === 'labeled' ? clonePolygon(row.polygon) : [];
      image = new Image();
      image.onload = () => {
        imageReady = true;
        fit();
        updateStats();
        const preload = selectedFrame(current, 1);
        if (preload !== current) {
          const img = new Image();
          img.src = `frame?frame=${preload}`;
        }
      };
      image.src = `frame?frame=${current}`;
      updateStats();
      draw();
    }

    async function savePolygon() {
      if (draft.length < 3) {
        setStatus('need 3 points', 'empty');
        return;
      }
      const res = await fetch('label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: current, polygon: draft })
      });
      state = await res.json();
      updateStats();
      draw();
      if (autoNextEl.checked) await loadFrame(selectedFrame(current, 1));
    }

    async function markNoComb() {
      const res = await fetch('no-comb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: current })
      });
      state = await res.json();
      draft = [];
      updateStats();
      draw();
      if (autoNextEl.checked) await loadFrame(selectedFrame(current, 1));
    }

    async function clearCurrent() {
      const res = await fetch('clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: current })
      });
      state = await res.json();
      draft = [];
      updateStats();
      draw();
    }

    function undoPoint() {
      draft.pop();
      updateStats();
      draw();
    }

    function addPoint(event) {
      if (!imageReady || dragMoved) return;
      const p = canvasPoint(event);
      const [ix, iy] = screenToImage(p.x, p.y);
      if (ix < 0 || iy < 0 || ix > image.naturalWidth || iy > image.naturalHeight) return;
      if (draft.length >= 3) {
        const [fx, fy] = imageToScreen(draft[0][0], draft[0][1]);
        const dist = Math.hypot(fx - p.x, fy - p.y);
        if (dist <= 14) {
          savePolygon();
          return;
        }
      }
      draft.push([Math.round(ix * 10) / 10, Math.round(iy * 10) / 10]);
      updateStats();
      draw();
    }

    canvas.addEventListener('mousemove', (event) => {
      const p = canvasPoint(event);
      lastMouse = p;
      if (dragging && dragStart) {
        const dx = p.x - dragStart.x;
        const dy = p.y - dragStart.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
        panX += dx;
        panY += dy;
        dragStart = p;
      }
      draw();
    });

    canvas.addEventListener('mouseleave', () => {
      lastMouse = null;
      draw();
    });

    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('mousedown', (event) => {
      if (event.button === 2 || event.shiftKey || event.altKey) {
        dragging = true;
        dragMoved = false;
        dragStart = canvasPoint(event);
        event.preventDefault();
      }
    });
    window.addEventListener('mouseup', () => {
      dragging = false;
      dragStart = null;
      setTimeout(() => { dragMoved = false; }, 0);
    });
    canvas.addEventListener('click', addPoint);
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const p = canvasPoint(event);
      zoomAt(event.deltaY < 0 ? 1.18 : 1 / 1.18, p.x, p.y);
    }, { passive: false });

    document.getElementById('prev').onclick = () => loadFrame(current - 1);
    document.getElementById('next').onclick = () => loadFrame(current + 1);
    document.getElementById('go').onclick = () => loadFrame(Number(jumpEl.value));
    document.getElementById('undo').onclick = undoPoint;
    document.getElementById('clear').onclick = clearCurrent;
    document.getElementById('noComb').onclick = markNoComb;
    document.getElementById('fit').onclick = fit;
    document.getElementById('zoomIn').onclick = () => zoomAt(1.35, canvas.width / 2, canvas.height / 2);
    document.getElementById('zoomOut').onclick = () => zoomAt(1 / 1.35, canvas.width / 2, canvas.height / 2);
    document.getElementById('save').onclick = savePolygon;
    jumpEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') loadFrame(Number(jumpEl.value));
    });

    window.addEventListener('keydown', async (event) => {
      if (event.target === jumpEl) return;
      if (event.key === 'ArrowRight' || event.key === ' ') await loadFrame(current + 1);
      else if (event.key === 'ArrowLeft') await loadFrame(current - 1);
      else if (event.key === 'Enter') await savePolygon();
      else if (event.key === 'Backspace' || event.key === 'u') undoPoint();
      else if (event.key === 'c') await clearCurrent();
      else if (event.key === 'n') await markNoComb();
      else if (event.key === 'f') fit();
      else if (event.key === '+' || event.key === '=') zoomAt(1.35, canvas.width / 2, canvas.height / 2);
      else if (event.key === '-' || event.key === '_') zoomAt(1 / 1.35, canvas.width / 2, canvas.height / 2);
      else return;
      event.preventDefault();
    });

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    loadState();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument(
        "--source-video",
        type=Path,
        default=None,
        help="Defaults to SAM2/data/videos/Best_Videos/<video_id>.mp4.",
    )
    parser.add_argument("--start-frame", type=int, default=DEFAULT_START_FRAME)
    parser.add_argument("--end-frame", type=int, default=DEFAULT_END_FRAME)
    parser.add_argument(
        "--frames",
        default=None,
        help="Comma-separated frame list/ranges, e.g. '18-30,87-90,97-115'.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-precache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def default_source_video(video_id: str) -> Path:
    return ROOT / "SAM2/data/videos/Best_Videos" / f"{video_id}.mp4"


def video_info(video_path: Path) -> dict[str, int | float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(text)
        tmp_name = f.name
    Path(tmp_name).replace(path)


def polygon_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(total) * 0.5


def polygon_bbox(poly: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def clean_polygon(poly: object, width: int, height: int) -> list[list[float]]:
    if not isinstance(poly, list):
        raise ValueError("polygon must be a list")
    cleaned: list[list[float]] = []
    for point in poly:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("polygon points must be [x, y]")
        x = float(point[0])
        y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon point must be finite")
        x = max(0.0, min(float(width - 1), x))
        y = max(0.0, min(float(height - 1), y))
        cleaned.append([round(x, 1), round(y, 1)])
    if len(cleaned) < 3:
        raise ValueError("polygon needs at least 3 points")
    return cleaned


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_frame_list(value: str | None, start_frame: int, end_frame: int) -> list[int]:
    if not value:
        return list(range(start_frame, end_frame + 1))
    frames: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid frame range: {item}")
            frames.update(range(start, end + 1))
        else:
            frames.add(int(item))
    if not frames:
        raise ValueError("--frames produced an empty frame list")
    return sorted(frames)


class LabelStore:
    def __init__(
        self,
        *,
        video_id: str,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        frame_indices: list[int],
        info: dict[str, int | float],
        labels_json: Path,
        labels_csv: Path,
    ) -> None:
        self.video_id = video_id
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_indices = sorted(dict.fromkeys(frame_indices))
        self.frame_set = set(self.frame_indices)
        self.info = info
        self.labels_json = labels_json
        self.labels_csv = labels_csv
        self.labels: dict[str, dict[str, object]] = {}
        if labels_json.exists():
            data = json.loads(labels_json.read_text(encoding="utf-8"))
            labels = data.get("labels", {})
            if isinstance(labels, dict):
                self.labels = labels
        self.save()

    def validate_frame(self, frame_index: int) -> None:
        if frame_index not in self.frame_set:
            raise ValueError(
                f"frame_index {frame_index} is not in the selected frame list"
            )

    def set_polygon(self, frame_index: int, poly: list[list[float]]) -> None:
        self.validate_frame(frame_index)
        cleaned = clean_polygon(
            poly,
            int(self.info["width"]),
            int(self.info["height"]),
        )
        area = polygon_area(cleaned)
        bbox = polygon_bbox(cleaned)
        self.labels[str(frame_index)] = {
            "frame_index": frame_index,
            "status": "labeled",
            "source": "manual_comb_polygon_tool",
            "polygon": cleaned,
            "bbox_xyxy": [round(x, 1) for x in bbox],
            "area_px2": round(area, 3),
            "area_cm2_sam2_0p02": round(area * CM_PER_PX_SAM2**2, 6),
            "updated_at": now_iso(),
        }
        self.save()

    def set_no_comb(self, frame_index: int) -> None:
        self.validate_frame(frame_index)
        self.labels[str(frame_index)] = {
            "frame_index": frame_index,
            "status": "no_comb",
            "source": "manual_comb_polygon_tool",
            "polygon": [],
            "updated_at": now_iso(),
        }
        self.save()

    def clear(self, frame_index: int) -> None:
        self.validate_frame(frame_index)
        self.labels.pop(str(frame_index), None)
        self.save()

    def state(self) -> dict[str, object]:
        visible_labels = {
            key: value
            for key, value in self.labels.items()
            if int(key) in self.frame_set
        }
        return {
            "video_id": self.video_id,
            "source_video": str(self.video_path),
            "frame_count": self.info["frame_count"],
            "fps": self.info["fps"],
            "width": self.info["width"],
            "height": self.info["height"],
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frame_indices,
            "labels_json": str(self.labels_json),
            "labels_csv": str(self.labels_csv),
            "labels": visible_labels,
        }

    def save(self) -> None:
        data = self.state()
        atomic_write_text(self.labels_json, json.dumps(data, indent=2))
        self.write_csv()

    def write_csv(self) -> None:
        self.labels_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "video_id",
            "frame_index",
            "status",
            "area_px2",
            "area_cm2_sam2_0p02",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "polygon_json",
            "source",
            "updated_at",
        ]
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=self.labels_csv.parent, delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame_index in self.frame_indices:
                label = self.labels.get(str(frame_index), {})
                bbox = label.get("bbox_xyxy") if label.get("status") == "labeled" else None
                row = {
                    "video_id": self.video_id,
                    "frame_index": frame_index,
                    "status": label.get("status", ""),
                    "area_px2": label.get("area_px2", ""),
                    "area_cm2_sam2_0p02": label.get("area_cm2_sam2_0p02", ""),
                    "bbox_x1": bbox[0] if isinstance(bbox, list) and len(bbox) == 4 else "",
                    "bbox_y1": bbox[1] if isinstance(bbox, list) and len(bbox) == 4 else "",
                    "bbox_x2": bbox[2] if isinstance(bbox, list) and len(bbox) == 4 else "",
                    "bbox_y2": bbox[3] if isinstance(bbox, list) and len(bbox) == 4 else "",
                    "polygon_json": json.dumps(label.get("polygon", []), separators=(",", ":")),
                    "source": label.get("source", ""),
                    "updated_at": label.get("updated_at", ""),
                }
                writer.writerow(row)
            tmp_name = f.name
        Path(tmp_name).replace(self.labels_csv)


def cached_frame_path(cache_dir: Path, frame_index: int) -> Path:
    return cache_dir / f"frame_{frame_index:06d}.jpg"


def ensure_frame_jpeg(video_path: Path, cache_dir: Path, frame_index: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = cached_frame_path(cache_dir, frame_index)
    if frame_path.exists():
        return frame_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame_index} from {video_path}")
        if not cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"Cannot write frame cache: {frame_path}")
    finally:
        cap.release()
    return frame_path


def precache_frames(video_path: Path, cache_dir: Path, frame_indices: list[int]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        frame_index
        for frame_index in frame_indices
        if not cached_frame_path(cache_dir, frame_index).exists()
    ]
    if not missing:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        for frame_index in missing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if not ok:
                raise ValueError(f"Cannot read frame {frame_index} from {video_path}")
            cv2.imwrite(
                str(cached_frame_path(cache_dir, frame_index)),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )
    finally:
        cap.release()


class LabelHandler(BaseHTTPRequestHandler):
    store: LabelStore
    video_path: Path
    frame_cache_dir: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: object, status: int = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(data).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def send_error_json(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8") or "{}")

    def route_path(self) -> str:
        path = urlparse(self.path).path
        for marker in ("/comb-label", "/comb-label/"):
            idx = path.find(marker)
            if idx >= 0:
                suffix = path[idx + len(marker) :]
                return "/" + suffix.lstrip("/")
        return path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = self.route_path()
        if route == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/state":
            self.send_json(self.store.state())
            return
        if route == "/frame":
            query = parse_qs(parsed.query)
            try:
                frame_index = int(query.get("frame", [""])[0])
                self.store.validate_frame(frame_index)
                frame_path = ensure_frame_jpeg(
                    self.video_path,
                    self.frame_cache_dir,
                    frame_index,
                )
                self.send_bytes(frame_path.read_bytes(), "image/jpeg")
            except Exception as exc:  # noqa: BLE001
                self.send_error_json(str(exc))
            return
        self.send_error_json("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = self.route_path()
        try:
            data = self.read_json()
            frame_index = int(data.get("frame_index", -1))
            if route == "/label":
                self.store.set_polygon(frame_index, data.get("polygon"))
                self.send_json(self.store.state())
                return
            if route == "/no-comb":
                self.store.set_no_comb(frame_index)
                self.send_json(self.store.state())
                return
            if route == "/clear":
                self.store.clear(frame_index)
                self.send_json(self.store.state())
                return
            if route == "/save":
                self.store.save()
                self.send_json(self.store.state())
                return
            self.send_error_json("not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self.send_error_json(str(exc))


def build_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    video_path = args.source_video or default_source_video(args.video_id)
    out_dir = args.output_root / args.video_id
    labels_json = out_dir / f"{args.video_id}_comb_manual_polygon_labels.json"
    labels_csv = out_dir / f"{args.video_id}_comb_manual_polygon_labels.csv"
    frame_cache_dir = out_dir / "frames"
    return video_path, labels_json, labels_csv, frame_cache_dir


def validate_range(start_frame: int, end_frame: int, frame_count: int) -> None:
    if start_frame < 0:
        raise ValueError("start-frame must be >= 0")
    if end_frame < start_frame:
        raise ValueError("end-frame must be >= start-frame")
    if end_frame >= frame_count:
        raise ValueError(
            f"end-frame {end_frame} outside video frame count {frame_count} "
            f"(last index {frame_count - 1})"
        )


def validate_frame_indices(frame_indices: list[int], frame_count: int) -> None:
    if not frame_indices:
        raise ValueError("No selected frames")
    bad = [frame for frame in frame_indices if frame < 0 or frame >= frame_count]
    if bad:
        raise ValueError(
            f"Selected frames outside video frame count {frame_count}: {bad}"
        )


def main() -> int:
    args = parse_args()
    video_path, labels_json, labels_csv, frame_cache_dir = build_paths(args)
    info = video_info(video_path)
    validate_range(args.start_frame, args.end_frame, int(info["frame_count"]))
    frame_indices = parse_frame_list(args.frames, args.start_frame, args.end_frame)
    validate_frame_indices(frame_indices, int(info["frame_count"]))
    store = LabelStore(
        video_id=args.video_id,
        video_path=video_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_indices=frame_indices,
        info=info,
        labels_json=labels_json,
        labels_csv=labels_csv,
    )

    if not args.no_precache:
        precache_frames(video_path, frame_cache_dir, frame_indices)

    print(f"video: {video_path}")
    print(
        f"frames: {args.start_frame}-{args.end_frame} / selected={len(frame_indices)} "
        f"/ total={info['frame_count']}"
    )
    print(f"selected frames: {','.join(str(x) for x in frame_indices)}")
    print(f"labels json: {labels_json}")
    print(f"labels csv: {labels_csv}")
    print(f"frames cache: {frame_cache_dir}")
    if args.prepare_only:
        print("prepared")
        return 0

    LabelHandler.store = store
    LabelHandler.video_path = video_path
    LabelHandler.frame_cache_dir = frame_cache_dir
    server = ThreadingHTTPServer((args.host, args.port), LabelHandler)
    print(f"open: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
