"""
red_line_endpoints.py
=====================

Detect red lines in an image (even noisy / jagged / crossing) and return
two endpoints per physical line segment, with coordinates expressed in a
center-origin system.

Usage:
    python red_line_endpoints.py path/to/your_image.png
"""

import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import atan2, pi

# ───────────────────────────── 1. 输入参数 ──────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python red_line_endpoints.py <image_file>")
    sys.exit(1)

IMG_PATH = Path(sys.argv[1])
assert IMG_PATH.exists(), f"File not found: {IMG_PATH}"

# ───────────────────────────── 2. 读图 & 红色掩膜 ───────────────────────
bgr = cv2.imread(str(IMG_PATH))
H, W = bgr.shape[:2]
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# HSV 红色有两段 (0-10°, 160-180°)
mask = cv2.inRange(hsv, (0, 100, 80),  (10, 255, 255))
mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

# 形态学清理：去掉散点、填小洞
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# ───────────────────────────── 3. Hough 取碎段 ──────────────────────────
segments = cv2.HoughLinesP(
    mask,                       # 输入二值图
    rho=1,                      # 像素分辨率
    theta=pi / 180,             # 角度分辨率 (1°)
    threshold=50,               # 检测阈值 (越低→段越多)
    minLineLength=80,           # 段最短长度
    maxLineGap=20               # 同一物理线允许的断裂
)

if segments is None:
    print("⚠ 未检测到任何线段")
    sys.exit(0)

segments = segments[:, 0]       # shape (N,4) → (x1,y1,x2,y2)

# ───────────────────────────── 4. (θ,ρ) 聚类 ───────────────────────────
tol_theta = 5 * pi / 180        # 角度容差 ±5°
tol_rho   = 20                  # 位置容差 ±20 px

groups = []                     # 每组 = [mean_theta, mean_rho, [segments...]]
for x1, y1, x2, y2 in segments:
    # ① 把段转为极坐标 (θ∈[0,π), ρ≥0)
    theta = atan2(y2 - y1, x2 - x1)
    if theta < 0:          # 方向无向，需要把 (–π,0] 映射到 [0,π)
        theta += pi
    # 单位法向量 n = (sinθ, –cosθ)，ρ = n·(x,y)
    n = np.array([math.sin(theta), -math.cos(theta)])
    rho = n.dot((x1, y1))

    # ② 尝试归到已有 group
    assigned = False
    for g in groups:
        mean_theta, mean_rho, segs = g
        dθ = min(abs(theta - mean_theta), pi - abs(theta - mean_theta))
        if dθ < tol_theta and abs(rho - mean_rho) < tol_rho:
            segs.append((x1, y1, x2, y2))
            # 更新 group 平均
            k = len(segs)
            g[0] = (mean_theta * (k - 1) + theta) / k
            g[1] = (mean_rho   * (k - 1) + rho)   / k
            assigned = True
            break
    if not assigned:
        groups.append([theta, rho, [(x1, y1, x2, y2)]])

# ───────────────────────────── 5. 每组合并端点 ──────────────────────────
cx_img, cy_img = W / 2.0, H / 2.0
results = []                    # [(x1,y1,x2,y2), ...]

for g_theta, g_rho, segs in groups:
    # 收集该物理红线的所有端点
    pts = np.array([(x1, y1) for x1, y1, _, _ in segs] +
                   [(x2, y2) for _, _, x2, y2 in segs])
    # 找欧氏距离最远的两个端点
    dist2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    p1, p2 = pts[i], pts[j]
    results.append((*p1, *p2))

# ───────────────────────────── 6. 输出结果 ──────────────────────────────
print(f"\n=== 检测到 {len(results)} 条红线，共 {2*len(results)} 个端点 ===")
print("坐标已换算到  (0,0) = 图像中心, 右 +X, 上 +Y\n")

for idx, (x1, y1, x2, y2) in enumerate(results, 1):
    print(f"Line {idx:02d}:",
          f"P1=({x1 - cx_img:+.1f}, {cy_img - y1:+.1f})",
          f"P2=({x2 - cx_img:+.1f}, {cy_img - y2:+.1f})")

# ───── 可选：把端点画到图上看一眼 ────────────────────────────────────────
SHOW = True          # 改为 False 可跳过显示
if SHOW:
    vis = bgr.copy()
    # 为不同物理线着不同颜色
    colors = [(0,255,0), (0,255,255), (255,0,0), (255,0,255),
              (0,128,255), (128,0,255), (255,128,0)]
    for idx, (x1,y1,x2,y2) in enumerate(results):
        col = colors[idx % len(colors)]
        cv2.circle(vis, (x1, y1), 7, col, -1)
        cv2.circle(vis, (x2, y2), 7, col, -1)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.title("Detected endpoints (color-coded)")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.show()
