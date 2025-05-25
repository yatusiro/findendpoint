from typing import List, Tuple
from pathlib import Path
from math import atan2, pi

import cv2
import numpy as np
from PIL import Image


def detect_red_line_endpoints(
    img_pil: Image.Image,
    *,
    hough_thresh: int = 30,
    min_len: int = 30,
    max_gap: int = 7,
    tol_theta_deg: float = 8.0,
    tol_rho_px: float = 25.0,
    margin_px: int = 5,
    dot_radius: int = 7,
    draw: bool = True,
) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
    """
    Detect red lines and return (annotated PIL image, endpoints list).

    Each endpoint tuple = (x1, y1, x2, y2) in centre-origin coords.
    Lines whose any endpoint lies within `margin_px` of the image border
    are discarded.
    """

    # 0. PIL → BGR
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # 1. 红色掩膜
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    # 2. Hough 碎段
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=pi / 180,
        threshold=hough_thresh,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if segments is None:
        return img_pil.copy() if draw else img_pil, []

    segs = segments[:, 0]              # (N,4)

    # 3. (θ,ρ) 聚类
    tol_theta = tol_theta_deg * pi / 180
    groups = []                        # [mean_theta, mean_rho, [segs]]
    for x1, y1, x2, y2 in segs:
        theta = atan2(y2 - y1, x2 - x1)
        if theta < 0:
            theta += pi
        rho = np.sin(theta) * x1 - np.cos(theta) * y1  # n·(x,y)

        for g in groups:
            mean_theta, mean_rho, gsegs = g
            if (min(abs(theta - mean_theta), pi - abs(theta - mean_theta)) < tol_theta
                    and abs(rho - mean_rho) < tol_rho_px):
                gsegs.append((x1, y1, x2, y2))
                # 更新均值
                k = len(gsegs)
                g[0] = (mean_theta * (k - 1) + theta) / k
                g[1] = (mean_rho   * (k - 1) + rho)   / k
                break
        else:
            groups.append([theta, rho, [(x1, y1, x2, y2)]])

    # 4. 每组取最远点对
    endpoints = []
    for _, _, gsegs in groups:
        pts = np.array([(x1, y1) for x1, y1, _, _ in gsegs] +
                       [(x2, y2) for _, _, x2, y2 in gsegs])
        if len(pts) < 2:
            continue
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        x1, y1 = pts[i]
        x2, y2 = pts[j]

        # 5. 距离边缘过滤
        if (x1 < margin_px or x1 > W - margin_px or
            y1 < margin_px or y1 > H - margin_px or
            x2 < margin_px or x2 > W - margin_px or
            y2 < margin_px or y2 > H - margin_px):
            continue

        endpoints.append((x1 - cx, cy - y1, x2 - cx, cy - y2))

    # 6. 绘制
    if not draw:
        return img_pil, endpoints

    vis = bgr.copy()
    blue = (255, 0, 0)                         # BGR → 蓝色
    for dx1, dy1, dx2, dy2 in endpoints:
        cv2.circle(vis, (int(dx1 + cx), int(cy - dy1)),
                   dot_radius, blue, -1)
        cv2.circle(vis, (int(dx2 + cx), int(cy - dy2)),
                   dot_radius, blue, -1)

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, endpoints



def drop_edge_endpoints(
    endpoints: list[tuple[float, float, float, float]],
    img_size: tuple[int, int],
    margin_px: int = 5
) -> list[tuple[float, float, float, float]]:
    """
    只删除端点本人靠近边缘的坐标，不影响同线另一端。

    Parameters
    ----------
    endpoints : list[(x1, y1, x2, y2)]
        *中心原点坐标系* 的端点列表。
    img_size : (width, height)
        图像分辨率，用于换算到左上原点坐标系。
    margin_px : int
        端点到四条边 < margin_px 像素 ⇒ 认为“贴边”并被清除。

    Returns
    -------
    new_endpoints : list[(x1, y1, x2, y2)]
        只保留 **不贴边** 的坐标；若某条线 2 端都贴边则整条被丢弃。
        若仅 1 端贴边，则返回 (x_good, y_good, x_good, y_good) ——仍占 4 位。
    """
    W, H = img_size
    cx, cy = W / 2.0, H / 2.0

    def is_edge(px, py) -> bool:
        # 判断左上原点坐标是否在边缘安全区外
        return (
            px < margin_px or px > W - margin_px or
            py < margin_px or py > H - margin_px
        )

    kept = []
    for x1, y1, x2, y2 in endpoints:
        # 还原到左上原点像素坐标
        ix1, iy1 = x1 + cx, cy - y1
        ix2, iy2 = x2 + cx, cy - y2

        # 判断
        p1_edge = is_edge(ix1, iy1)
        p2_edge = is_edge(ix2, iy2)

        if p1_edge and p2_edge:
            # 两端都贴边 ⇒ 丢整条
            continue
        elif p1_edge:
            # 只保留 p2，以占位方式存两次
            kept.append((x2, y2, x2, y2))
        elif p2_edge:
            # 只保留 p1
            kept.append((x1, y1, x1, y1))
        else:
            # 两端都合格
            kept.append((x1, y1, x2, y2))

    return kept




# ───────────── quick test ─────────────
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open("test.jpg")
    annot, pts = detect_red_line_endpoints(img)

    print(f"保留下 {len(pts)} 条线 ({2*len(pts)} 端点)：")
    for i, (x1, y1, x2, y2) in enumerate(pts, 1):
        print(f"{i:02d}: P1=({x1:+.1f},{y1:+.1f})  P2=({x2:+.1f},{y2:+.1f})")

    plt.imshow(annot)
    plt.axis('off')
    plt.show()








from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def endpoints_skeleton_dbscan(
    img_pil: Image.Image,
    *,
    line_width: int = 15,      # 估计的红线宽度，用于 DBSCAN eps
    min_obj: int = 50,         # 小于此骨架像素数的连通域视为噪声
    dot_radius: int = 6        # 端点可视化圆半径
) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
    """
    Detect red-line endpoints via skeleton + DBSCAN deduplication.

    Parameters
    ----------
    img_pil : PIL.Image
    line_width : int
        Approximate physical red-line thickness in pixels.
    min_obj : int
        Connected-component size below which skeleton blobs are ignored.
    dot_radius : int
        Radius of green endpoint dots in the output image.

    Returns
    -------
    annot_pil : PIL.Image
        RGB image with endpoints drawn.
    endpoints : list[tuple]
        (x1, y1, x2, y2) for each physical line, expressed in a
        centre-origin coordinate system (right +X, up +Y).
    """

    # ------ 0. PIL → OpenCV BGR ------
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # ------ 1. 提取红色掩膜 ------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, 1)

    # ------ 2. 骨架化 ------
    skel = skeletonize(mask > 0).astype(np.uint8)

    # ------ 3. 骨架端点（度 = 1）------
    coords = np.column_stack(np.where(skel))          # (y, x)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    pix2i = {tuple(p): i for i, p in enumerate(coords)}
    deg = np.zeros(len(coords), int)
    for idx, (y, x) in enumerate(coords):
        for dy, dx in nbrs:
            if (y+dy, x+dx) in pix2i:
                deg[idx] += 1
    endpts = coords[deg == 1]                         # (y, x)

    if len(endpts) == 0:
        return img_pil.copy(), []

    # ------ 4. DBSCAN 去重 ------
    eps = max(2, line_width / 2)
    db = DBSCAN(eps=eps, min_samples=1).fit(endpts)
    centers = np.array(
        [endpts[db.labels_ == lb].mean(axis=0)
         for lb in np.unique(db.labels_)]
    )                                                # (cy, cx)

    # ------ 5. 连通域分组 & 取最远两点 ------
    _, comp_lab = cv2.connectedComponents(skel, 8)
    groups = {}
    for cy0, cx0 in centers:
        lbl = comp_lab[int(cy0), int(cx0)]
        groups.setdefault(lbl, []).append((cx0, cy0))  # (x, y)

    endpoints = []
    for pts in groups.values():
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        endpoints.append((x1 - cx, cy - y1, x2 - cx, cy - y2))

    # ------ 6. 可视化 ------
    vis = bgr.copy()
    for x1_, y1_, x2_, y2_ in endpoints:
        cv2.circle(vis, (int(x1_ + cx), int(cy - y1_)),
                   dot_radius, (0, 255, 0), -1)
        cv2.circle(vis, (int(x2_ + cx), int(cy - y2_)),
                   dot_radius, (0, 255, 0), -1)

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, endpoints


