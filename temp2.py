from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from math import atan2, pi

def detect_valid_endpoints(self, img_pil: Image.Image) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    只检测满足条件的红色线段端点对（已排除倾斜、上下边缘、聚类处理）。
    返回每组中最远的一对端点（图像中心为原点坐标系）。
    """
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W // 2.0, H // 2.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)

    segments = cv2.HoughLinesP(mask, 1, np.pi / 180,
                               threshold=self.hough_thresh,
                               minLineLength=self.min_len,
                               maxLineGap=self.max_gap)
    if segments is None:
        return []

    segs = segments[:, 0]
    groups = []
    for x1, y1, x2, y2 in segs:
        theta = atan2(y2 - y1, x2 - x1)
        if theta < 0:
            theta += pi
        n = np.array([np.sin(theta), -np.cos(theta)])
        rho = n.dot((x1, y1))
        for g in groups:
            mean_theta, mean_rho, gsegs = g
            dth = min(abs(theta - mean_theta), pi - abs(theta - mean_theta))
            if dth < self.tol_theta and abs(rho - mean_rho) < self.tol_rho_px:
                gsegs.append((x1, y1, x2, y2))
                k = len(gsegs)
                g[0] = (mean_theta * (k - 1) + theta) / k
                g[1] = (mean_rho * (k - 1) + rho) / k
                break
        else:
            groups.append([theta, rho, [(x1, y1, x2, y2)]])

    top_band = self.edge_pct * H
    bottom_band = (1 - self.edge_pct) * H

    result_pairs = []

    for _, _, gsegs in groups:
        pts = np.array([(x1, y1) for x1, y1, _, _ in gsegs] +
                       [(x2, y2) for _, _, x2, y2 in gsegs])
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        (x1, y1), (x2, y2) = pts[i], pts[j]

        angle_deg = abs(atan2(y2 - y1, x2 - x1)) * 180 / pi
        if angle_deg > self.angle_thr_deg:
            continue
        if (y1 < top_band and y2 < top_band) or (y1 > bottom_band and y2 > bottom_band):
            continue

        pt1 = (int(round(x1 - cx)), int(round(cy - y1)))
        pt2 = (int(round(x2 - cx)), int(round(cy - y2)))
        result_pairs.append((pt1, pt2))

    return result_pairs
