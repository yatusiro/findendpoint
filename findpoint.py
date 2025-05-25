"""
skeleton_dbscan_endpoints.py
────────────────────────────
Detect endpoints of red lines using
  • skeletonization  (skimage)
  • degree-1 pixel detection
  • DBSCAN clustering to merge duplicates
Outputs 2×N endpoints and shows them in green.
"""

import math, cv2, numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from pathlib import Path

# ─────────────────── 1. 读取图像 ───────────────────
# if len(sys.argv) < 2:
#     print("Usage: python skeleton_dbscan_endpoints.py <image>")
#     sys.exit(1)

# img = cv2.imread(sys.argv[1])
# if img is None:
#     raise FileNotFoundError(sys.argv[1])
# H, W = img.shape[:2]
# cx, cy = W / 2.0, H / 2.0       # 中心 ←→ (0,0)
IMG_PATH = Path(__file__).with_name("test.jpg")   # 与脚本同目录
if not IMG_PATH.exists():
    raise FileNotFoundError(f"找不到 {IMG_PATH}")

img = cv2.imread(str(IMG_PATH))
if img is None:
    raise RuntimeError("OpenCV 读取失败，请检查文件格式/路径")
H, W = img.shape[:2]
cx, cy = W / 2.0, H / 2.0

# ─────────────────── 2. 提取红色掩膜 ────────────────
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0,100,80),  (10,255,255))
mask |= cv2.inRange(hsv, (160,100,80), (180,255,255))

ker  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)

# ─────────────────── 3. 骨架化 ─────────────────────
skel = skeletonize(mask > 0).astype(np.uint8)

# ─────────────────── 4. 找骨架端点 (度=1) ───────────
coords = np.column_stack(np.where(skel))      # (y,x)
nbrs  = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
pix2i = {tuple(p): i for i, p in enumerate(coords)}
deg   = np.zeros(len(coords), int)

for idx, (y,x) in enumerate(coords):
    for dy,dx in nbrs:
        if (y+dy, x+dx) in pix2i:
            deg[idx] += 1

endpts = coords[deg == 1]                     # (y,x)

# ─────────────────── 5. DBSCAN 去重 ────────────────
line_width = 15           # ≈线宽；可按实际情况调整
clu  = DBSCAN(eps=line_width/2, min_samples=1).fit(endpts)
labels = clu.labels_
centers = np.array([endpts[labels==lb].mean(axis=0)
                    for lb in np.unique(labels)])   # (cy,cx)

# ─────────────────── 6. 每根线取 2 端点 ─────────────
# 先对骨架做连通域 → 把端点按 label 分组
_, comp_lab = cv2.connectedComponents(skel, connectivity=8)
lines = {}
for (cy0,cx0) in centers:
    lbl = comp_lab[int(cy0), int(cx0)]
    lines.setdefault(lbl, []).append((cx0, cy0))    # (x,y)

results = []          # [(x1,y1,x2,y2), ...]
for lbl, pts in lines.items():
    if len(pts) < 2:
        continue
    # 取组内最远两点
    pts = np.array(pts)
    d2  = np.sum((pts[:,None]-pts[None,:])**2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    x1,y1 = pts[i];  x2,y2 = pts[j]
    results.append((x1,y1,x2,y2))

# ─────────────────── 7. 输出 & 可视化 ──────────────
print(f"\n检测到 {len(results)} 条红线 {2*len(results)} 端点：\n")
for k,(x1,y1,x2,y2) in enumerate(results,1):
    print(f"Line {k:02d}:",
          f"P1=({x1-cx:+.1f}, {cy-y1:+.1f})",
          f"P2=({x2-cx:+.1f}, {cy-y2:+.1f})")

# 绿色端点可视化
vis = img.copy()
for x1,y1,x2,y2 in results:
    cv2.circle(vis, (int(x1),int(y1)), 6, (0,255,0), -1)
    cv2.circle(vis, (int(x2),int(y2)), 6, (0,255,0), -1)

plt.figure(figsize=(6,6))
plt.axis('off')
plt.title("Deduplicated Endpoints (Green)")
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.show()
