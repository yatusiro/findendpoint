import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def detect_insertions(
        img_path: str | Path,
        *,
        canny1: int = 50,
        canny2: int = 150,
        hough_thresh: int = 40,      # ↓ 适度降低
        min_len: int = 60,           # ↓ 允许更短的线段
        max_gap: int = 25,
        angle_tol_deg: float = 6.0,  # ↑ 放宽角度容差
        rho_tol_px: float = 15.0,    # ↑ 放宽距离容差
        min_cluster_height: int = 80,# ★聚类后，总垂直跨度须≥该值
        merge_dist_px: int = 8,
        dot_r: int = 4,
        bgr_dot: tuple[int,int,int] = (0,0,255),
):
    """
    返回 (坐标列表, 标注后图像)；坐标系左下为 (0,0)。
    """
    img_path = Path(img_path)
    img  = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) Canny
    edges = cv2.Canny(gray, canny1, canny2, apertureSize=3)

    # 2) 概率霍夫
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    if lines is None:
        return [], img.copy()

    # 3) 用 (ρ,θ) 聚类
    buckets = defaultdict(list)
    for x1,y1,x2,y2 in lines[:,0]:
        theta = np.mod(np.arctan2(y2-y1, x2-x1), np.pi)
        deg   = np.degrees(theta)
        if not (15 < deg < 75):  # 依然过滤水平垂直
            continue
        rho = x1*np.cos(theta) + y1*np.sin(theta)
        key = (round(theta/np.deg2rad(angle_tol_deg)),
               round(rho/rho_tol_px))
        buckets[key].append(((x1,y1),(x2,y2)))

    endpoints, cluster_heights = [], []
    for segs in buckets.values():
        pts = [p for s in segs for p in s]
        ys  = [p[1] for p in pts]
        height = max(ys) - min(ys)  # 该簇的总垂直跨度
        cluster_heights.append(height)
        if height < min_cluster_height:   # ★太短 → 噪声
            continue
        # 插入点 = y 最大的端点
        endpoints.append(max(pts, key=lambda p: p[1]))

    if not endpoints:
        return [], img.copy()

    # 4) 距离再次合并
    P      = np.array(endpoints, float)
    merged = []
    used   = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        if used[i]: continue
        close = np.linalg.norm(P - p, axis=1) < merge_dist_px
        group = P[close]
        used[close] = True
        merged.append(group.mean(axis=0))
    merged = np.array(merged, int)

    # 5) 标注与坐标转换
    annot, coords = img.copy(), []
    for x, y in merged:
        cv2.circle(annot, (x,y), dot_r, bgr_dot, -1)
        coords.append((int(x), int(h-1-y)))  # 左下原点

    return coords, annot


# 批量示例
if __name__ == "__main__":
    src = Path("./samples")
    dst = Path("./annotated")
    dst.mkdir(exist_ok=True)

    for f in src.glob("*.jpg"):
        pts, out = detect_insertions(f)
        print(f"{f.name}: {len(pts)} 点 ->", pts)
        cv2.imwrite(str(dst/f.name), out)
