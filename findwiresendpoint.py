import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def detect_insertions(
        img_path: str | Path,
        *,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
        hough_thresh: int = 60,
        min_line_len: int = 120,       # ★直线最短长度阈值（像素）
        max_line_gap: int = 20,        # 同一直线段允许的最大间断
        angle_tol_deg: float = 4.0,    # (ρ,θ) 聚类时的角度容差
        rho_tol_px: float = 12.0,      # (ρ,θ) 聚类时的距离容差
        dot_radius: int = 4,           # 红点半径
        bgr_dot: tuple[int,int,int] = (0, 0, 255),
        merge_dist_px: int = 6,        # 近邻端点再次合并阈值
):
    """
    检测斜向线缆的“插入端”，并返回坐标与标注后图像。

    返回
    ----
    coords : list[(x, y)]
        以图片左下角为 (0,0) 的插入点坐标。
    annot  : np.ndarray
        已画出红点的 BGR 图像。
    """
    img_path = Path(img_path)
    img   = cv2.imread(str(img_path))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape

    # 1) Canny 边缘检测
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    # 2) 概率霍夫直线检测
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return [], img.copy()

    # 3) 用 (ρ,θ) 对直线进行粗聚类，归并同一根线缆
    buckets = defaultdict(list)              # key -> 该线缆的所有线段
    for x1, y1, x2, y2 in lines[:, 0]:
        theta = np.mod(np.arctan2(y2 - y1, x2 - x1), np.pi)  # 角度 ∈ [0, π)
        deg   = np.degrees(theta)
        if not (15 < deg < 75):              # 过滤掉水平或垂直干扰
            continue

        rho = x1 * np.cos(theta) + y1 * np.sin(theta)        # 直线极坐标 ρ
        key = (round(theta / np.deg2rad(angle_tol_deg)),
               round(rho / rho_tol_px))
        buckets[key].append(((x1, y1), (x2, y2)))

    # 4) 对每个线缆聚类，仅保留最靠下（y 最大）的端点
    endpoints = []
    for segs in buckets.values():
        all_pts = [p for seg in segs for p in seg]
        px, py = max(all_pts, key=lambda p: p[1])  # y 最大即最底端
        endpoints.append((px, py))

    if not endpoints:
        return [], img.copy()

    # 5) 再次合并彼此过近的端点（避免重复）
    pts    = np.array(endpoints, dtype=float)
    merged = []
    used   = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        dist    = np.linalg.norm(pts - p, axis=1)
        for j in np.where((dist < merge_dist_px) & (~used))[0]:
            cluster.append(pts[j]); used[j] = True
        merged.append(np.mean(cluster, axis=0))
    merged = np.array(merged, dtype=int)

    # 6) 在图上画红点，并转换坐标系
    annot  = img.copy()
    coords = []
    for x, y in merged:
        cv2.circle(annot, (x, y), dot_radius, bgr_dot, -1)
        coords.append((int(x), int(h - 1 - y)))   # 翻转 y 轴

    return coords, annot


# ----------------- 批量处理示例 -----------------
if __name__ == "__main__":
    src_dir = Path("./samples")     # 输入文件夹
    dst_dir = Path("./annotated2")  # 输出文件夹
    dst_dir.mkdir(exist_ok=True)

    for img_file in src_dir.glob("*.jpg"):
        pts, annotated_img = detect_insertions(img_file)
        print(img_file.name, pts)
        cv2.imwrite(str(dst_dir / img_file.name), annotated_img)
