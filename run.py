
import cv2  
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image, ImageEnhance
from io import BytesIO
import requests
from pathlib import Path
from typing import List, Tuple
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from math import atan2, pi

# path = "MR1 BP_1_20250516_135217.jpg"

def detect_and_highlight_wires_test(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    # result_image.show()
    return result_image

def find_endpoints(
        img_pil: Image.Image,
        *,
        # ---- Hough 参数 ----
        hough_thresh: int = 70,      # 至少多少投票才算一条线
        min_len: int = 60,           # 检测到的线段最短长度 (px)
        max_gap: int = 4,            # 允许的断裂空隙 (px)
        # ---- 共线聚类容忍 ----
        tol_theta_deg: float = 2.0,  # 角度差容忍 (°)
        tol_rho_px: float = 10.0,    # ρ 差容忍 (px)
        # ---- 端点可视化 ----
        dot_radius: int = 7,         # 端点圆半径 (px)
        # ---- 业务过滤阈值 ----
        angle_thr_deg: float = 65.0, # 只要倾角 ≤ 65° 的线
        edge_pct: float = 0.20,      # 上/下黑影区域百分比
        side_pct: float = 0.12       # 左右边缘区百分比
) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
    """
    在红色线段上找端点，只保留
      1) 倾角 ≤ angle_thr_deg
      2) 端点对不同时落在上下 edge_pct
      3) 恰好一个端点落在左右 side_pct 边缘区
    返回 (带蓝点的可视化图, 端点坐标列表)
    """

    # --- PIL → BGR ---
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W // 2.0, H // 2.0

    # --- 提取红色掩膜 ---
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, (0, 100, 80),  (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    # 如需更严格断开，可注释掉下行闭运算
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    # --- 概率 HoughLinesP ---
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    if segments is None:
        return img_pil.copy(), []

    segs = segments[:, 0]  # (N, 4)

    # --- (θ,ρ) 聚类，合并同一直线 ---
    tol_theta = tol_theta_deg * pi / 180
    groups = []
    for x1, y1, x2, y2 in segs:
        theta = atan2(y2 - y1, x2 - x1)
        if theta < 0:
            theta += pi                     # 保证 [0, π)
        n = np.array([np.sin(theta), -np.cos(theta)])
        rho = n.dot((x1, y1))

        for g in groups:
            mean_theta, mean_rho, gsegs = g
            dth = min(abs(theta - mean_theta), pi - abs(theta - mean_theta))
            if dth < tol_theta and abs(rho - mean_rho) < tol_rho_px:
                gsegs.append((x1, y1, x2, y2))
                k = len(gsegs)
                g[0] = (mean_theta * (k - 1) + theta) / k
                g[1] = (mean_rho  * (k - 1) + rho)   / k
                break
        else:
            groups.append([theta, rho, [(x1, y1, x2, y2)]])

    # --- 三道业务筛选 ---
    endpoints, keep_pairs = [], []
    top_band    = edge_pct * H
    bottom_band = (1 - edge_pct) * H
    left_thr    = side_pct * W
    right_thr   = (1 - side_pct) * W

    for _, _, gsegs in groups:
        # 取最远两点作为该直线的端点
        pts = np.array([(x1, y1) for x1, y1, _, _ in gsegs] +
                       [(x2, y2) for _, _, x2, y2 in gsegs])
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        (x1, y1), (x2, y2) = pts[i], pts[j]

        # (1) 倾角过滤
        angle_deg = abs(atan2(y2 - y1, x2 - x1)) * 180 / pi
        if angle_deg > angle_thr_deg:
            continue

        # (2) 上下黑影过滤：整对端点落同一黑影区则丢弃
        if (y1 < top_band and y2 < top_band) or (y1 > bottom_band and y2 > bottom_band):
            continue

        # (3) 左右边缘过滤：恰好一个端点在边缘区
        in_left  = (x1 <= left_thr)  or (x2 <= left_thr)
        in_right = (x1 >= right_thr) or (x2 >= right_thr)
        if not (in_left ^ in_right):   # ^ 为异或：只能有一个 True
            continue

        # 满足全部条件 → 保留
        endpoints.append((x1 - cx, cy - y1, x2 - cx, cy - y2))
        keep_pairs.append(((x1, y1), (x2, y2)))

    # --- 画端点 ---
    vis = bgr.copy()
    for (p1, p2) in keep_pairs:
        for (x, y) in (p1, p2):
            cv2.circle(vis, (int(x), int(y)),
                       dot_radius, (255, 0, 0), -1)  # 纯蓝圆点

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, endpoints


folder_path = "segment/boat"

jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
print(jpg_files)
for path in jpg_files:
    print(path)
    img = Image.open(path)
    processed_img = detect_and_highlight_wires_test(img)
    annotated, pts = find_endpoints(processed_img)
    outputfloder = "endpoint"
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(outputfloder, f"{name}_processed{ext}")
    annotated.save(output_path)








def find_endpoints_init(img_pil: Image.Image,
                  *,
                  houth_thresh: int = 25, #50 (threshold) 直线投票值，越大越严格
                  max_len: int = 15, #80 (minLineLength) 路线最短长度
                  max_gap: int = 7, #20 (maxLineGap) 同一条直线内可容忍的最大间隔
                  tol_theta_deg: float = 5.0, # # 角度容忍度，单位为度
                  tol_rho_px: float = 20.0,  # 距离容忍度，单位为像素
                  dot_radius: int = 7,) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:

    # PIL->OpenCV BGR
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W // 2.0, H // 2.0

    # red mask
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #(5, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    # Hough Transform
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=houth_thresh,
        minLineLength=max_len,
        maxLineGap=max_gap
    )

    if segments is None:
        return img_pil.copy(), []
    
    segs = segments[:, 0]

    # (theta, rho) clustering
    tol_theta = tol_theta_deg * pi / 180
    groups = []
    for x1, y1, x2, y2 in segs:
        theta = atan2(y2 - y1, x2 - x1)
        if theta < 0:
            theta += pi
        n = np.array([np.sin(theta), -np.cos(theta)])
        rho = n.dot((x1, y1))

        added = False

        for g in groups:
            mean_theta, mean_rho, gsegs =g
            dtheata = min(abs(theta - mean_theta), pi - abs(theta - mean_theta))
            if dtheata < tol_theta and abs(rho - mean_rho) < tol_rho_px:
                gsegs.append((x1, y1, x2, y2))
                k = len(gsegs)
                g[0] = (mean_theta * (k - 1) + theta) / k
                g[1] = (mean_rho * (k - 1) + rho) / k
                added = True
                break
        if not added:
            groups.append([theta, rho, [(x1, y1, x2, y2)]])

    endpoints = []
    for _,_, gsegs in groups:
        pts = np.array([(x1, y1) for x1, y1, _, _ in gsegs] + [(x2, y2) for _, _, x2, y2 in gsegs])
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        (x1, y1), (x2, y2) = pts[i], pts[j]
        endpoints.append((x1 - cx, cy -y1, x2 - cx, cy - y2))


    

    # draw endpoints    
    # colours = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255), (255, 128, 0)]

    vis = bgr.copy()
    for idx, (x1_, y1_, x2_, y2_) in enumerate(endpoints):
        # col = colours[idx % len(colours)]
        col = (255, 0, 0)
        cv2.circle(vis, (int(x1_ + cx), int(cy - y1_)), dot_radius, col, -1)
        cv2.circle(vis, (int(x2_ + cx), int(cy - y2_)), dot_radius, col, -1)

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, endpoints

def find_endpoints_after(
        img_pil: Image.Image,
        *,
        hough_thresh: int = 70,
        min_len: int = 60,
        max_gap: int = 4,
        tol_theta_deg: float = 2.0,
        tol_rho_px: float = 10.0,
        dot_radius: int = 7,
        angle_thr_deg: float = 65.0,   # ★ 新增：最大倾角
        edge_pct: float = 0.20,        # ★ 新增：上下黑影百分比
) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:

    # --- 基础准备 ---
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W // 2.0, H // 2.0

    # --- 提取红色掩膜 ---
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    # morphologyEx(mask, MORPH_CLOSE, ...) 如需更严格断口，可注释掉或缩小核

    # --- Hough 直线检测 ---
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    if segments is None:
        return img_pil.copy(), []

    segs = segments[:, 0]  # (N,4)

    # --- (θ,ρ) 聚类，合并共线短段 ---
    tol_theta = tol_theta_deg * np.pi / 180
    groups = []
    for x1, y1, x2, y2 in segs:
        theta = atan2(y2 - y1, x2 - x1)
        if theta < 0:                # 保证范围 [0,π)
            theta += np.pi
        n = np.array([np.sin(theta), -np.cos(theta)])
        rho = n.dot((x1, y1))

        for g in groups:
            mean_theta, mean_rho, gsegs = g
            dth = min(abs(theta - mean_theta), np.pi - abs(theta - mean_theta))
            if dth < tol_theta and abs(rho - mean_rho) < tol_rho_px:
                gsegs.append((x1, y1, x2, y2))
                k = len(gsegs)
                g[0] = (mean_theta * (k - 1) + theta) / k
                g[1] = (mean_rho  * (k - 1) + rho)   / k
                break
        else:
            groups.append([theta, rho, [(x1, y1, x2, y2)]])

    endpoints = []
    keep_pairs = []  # 仅用于可视化
    top_band    = edge_pct * H           # ★ 上黑影阈值
    bottom_band = (1 - edge_pct) * H     # ★ 下黑影阈值

    for _, _, gsegs in groups:
        pts = np.array([(x1, y1) for x1, y1, _, _ in gsegs] +
                       [(x2, y2) for _, _, x2, y2 in gsegs])
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        (x1, y1), (x2, y2) = pts[i], pts[j]

        # ---------- 额外两重筛选 ----------
        # ① 倾角限制
        angle_deg = abs(atan2(y2 - y1, x2 - x1)) * 180 / pi
        if angle_deg > angle_thr_deg:
            continue

        # ② 位于上下黑影区的整对端点剔除
        if (y1 < top_band and y2 < top_band) or (y1 > bottom_band and y2 > bottom_band):
            continue
        # ----------------------------------

        endpoints.append((x1 - cx, cy - y1, x2 - cx, cy - y2))
        keep_pairs.append(((x1, y1), (x2, y2)))

    # --- 结果可视化 ---
    vis = bgr.copy()
    for (p1, p2) in keep_pairs:
        for (x, y) in (p1, p2):
            cv2.circle(vis, (int(x), int(y)), dot_radius, (255, 0, 0), -1)

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, endpoints
