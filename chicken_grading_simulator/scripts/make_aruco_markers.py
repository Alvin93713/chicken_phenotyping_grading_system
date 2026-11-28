"""A4 printable ArUco calibration markers (DICT_4X4_50) at exact physical size, 600 DPI."""
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600
cm = lambda v: int(round(v / 2.54 * DPI))
W, H = cm(21.0), cm(29.7)
out = sys.argv[1]
page = Image.new("L", (W, H), 255)
d = ImageDraw.Draw(page)
F = "C:/Windows/Fonts/msjh.ttc"; FB = "C:/Windows/Fonts/msjhbd.ttc"
font = lambda size, bold=False: ImageFont.truetype(FB if bold else F, size)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

y = cm(1.3)
d.text((cm(1.5), y), "雞隻表型分級系統：ArUco 比例尺校正標記", font=font(cm(0.62), True), fill=0)
y += cm(1.0)
for line in ("列印時請選「實際大小」或「100%」，不要勾選「符合頁面大小」。",
             "印好後先用尺量下方 10 cm 檢查線，誤差超過 1 mm 請重新設定印表機後再印。",
             "ID 0 → 雞冠相機　　ID 1 → 腳脛相機。每台相機的畫面中只放一張自己的標記。",
             "標記尺寸 = 黑色外框的邊長；請保留四周白邊，貼在平整的硬板上，與雞冠／腳脛同一平面。"):
    d.text((cm(1.5), y), line, font=font(cm(0.36)), fill=0); y += cm(0.58)

# 10 cm check line
y += cm(0.4); x0 = cm(1.5)
d.line((x0, y + cm(0.5), x0 + cm(10), y + cm(0.5)), fill=0, width=6)
for i in range(11):
    h = cm(0.5) if i in (0, 5, 10) else cm(0.3)
    x = x0 + cm(i)
    d.line((x, y + cm(0.5) - h, x, y + cm(0.5)), fill=0, width=5)
d.text((x0 + cm(10.4), y + cm(0.05)), "10 cm 檢查線", font=font(cm(0.38), True), fill=0)
y += cm(1.4)

def dashed_rect(x1, y1, x2, y2, dash=cm(0.25)):
    for a, b, c, e in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x1, y2, x2, y2), (x1, y1, x1, y2)):
        length = max(abs(c - a), abs(e - b)); n = int(length // dash)
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1, (k + 1) / n)
            d.line((a + (c - a) * t0, b + (e - b) * t0, a + (c - a) * t1, b + (e - b) * t1), fill=110, width=3)

placed = []
for size in (5.0, 4.0, 3.0):
    x = cm(1.5)
    margin = cm(0.8)  # white quiet zone around the marker (inside the cut line)
    for marker_id, name in ((0, "雞冠"), (1, "腳脛")):
        side = cm(size)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=1)
        cut = (x, y, x + side + 2 * margin, y + side + 2 * margin + cm(0.75))
        dashed_rect(*cut)
        page.paste(Image.fromarray(img), (x + margin, y + margin))
        d.text((x + margin, y + margin + side + cm(0.15)), f"ID {marker_id}　{name}　{size:.0f} cm",
               font=font(cm(0.36), True), fill=0)
        placed.append((marker_id, size))
        x = cut[2] + cm(0.6)
    y += cm(size) + 2 * margin + cm(0.75) + cm(0.45)

d.text((cm(1.5), H - cm(1.2)), "字典 DICT_4X4_50 ｜ 600 DPI ｜ A4", font=font(cm(0.3)), fill=90)
page.convert("RGB").save(out, "PDF", resolution=DPI)
page.save(out.replace(".pdf", "_preview.png"))

# self-check: detect every marker on the rendered page and measure its side length
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
gray = np.array(page)
corners, ids, _ = detector.detectMarkers(gray)
for c, i in sorted(zip(corners, ids.flatten()), key=lambda p: (p[0][0][:, 1].min(), p[0][0][:, 0].min())):
    pts = c[0]
    sides = [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
    print(f"ID {i}: side {np.mean(sides) / DPI * 2.54:.3f} cm")
print("placed", placed, "page", W, H, "y_end_cm", round(y / DPI * 2.54, 1))
