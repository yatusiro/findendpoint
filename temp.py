import os
import glob
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
from math import atan2, pi

class ProbabilisticHough:
    def __init__(
        self,
        hough_thresh=10,
        min_len=18,
        max_gap=4,
        tol_theta_deg=5.0,
        tol_rho_px=10.0,
        dot_radius=7,
        angle_thr_deg=25.0,
        edge_pct=0.20,
        side_pct=0.12,
    ):
        self.hough_thresh = hough_thresh
        self.min_len = min_len
        self.max_gap = max_gap
        self.tol_theta = tol_theta_deg * pi / 180
        self.tol_rho_px = tol_rho_px
        self.dot_radius = dot_radius
        self.angle_thr_deg = angle_thr_deg
        self.edge_pct = edge_pct
        self.side_pct = side_pct

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    def detect_endpoints(self, img_pil: Image.Image) -> Tuple[bool, Tuple[int, int], Tuple[int, int]]:
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        cx, cy = W // 2.0, H // 2.0

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
        mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)

        segments = cv2.HoughLinesP(mask, 1, np.pi / 180,
                                   threshold=self.hough_thresh,
                                   minLineLength=self.min_len,
                                   maxLineGap=self.max_gap)
        if segments is None:
            return False, (None, None), (None, None)

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
        left_thr = self.side_pct * W
        right_thr = (1 - self.side_pct) * W

        closest_point = None
        other_point = None
        min_dist = float('inf')

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
            in_left = (x1 <= left_thr) or (x2 <= left_thr)
            in_right = (x1 >= right_thr) or (x2 >= right_thr)
            if not (in_left ^ in_right):
                continue

            pt1 = (x1 - cx, cy - y1)
            pt2 = (x2 - cx, cy - y2)

            for a, b in [(pt1, pt2), (pt2, pt1)]:
                if abs(a[0]) > (self.side_pct * W / 2):
                    dist = a[0] ** 2 + a[1] ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = a
                        other_point = b

        if closest_point:
            return True, (int(round(closest_point[0])), int(round(closest_point[1]))), \
                         (int(round(other_point[0])), int(round(other_point[1])))
        else:
            return False, (None, None), (None, None)

    def detect(self, image: Image.Image) -> Tuple[bool, Tuple[int, int], Tuple[int, int]]:
        processed = self.preprocess_image(image)
        return self.detect_endpoints(processed)

if __name__ == "__main__":
    os.makedirs("output2", exist_ok=True)
    detector = ProbabilisticHough()

    image_paths = glob.glob("segment/boat/*.jpg")
    for path in image_paths:
        image = Image.open(path)
        success, pt1, pt2 = detector.detect(image)

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        save_path = os.path.join("output2", os.path.basename(path))

        if success:
            print(f"{path} → 检测成功: {pt1}, {pt2}")
            H, W = img_cv.shape[:2]
            cx, cy = W // 2, H // 2
            x1, y1 = int(cx + pt1[0]), int(cy - pt1[1])
            x2, y2 = int(cx + pt2[0]), int(cy - pt2[1])
            cv2.circle(img_cv, (x1, y1), 6, (0, 0, 255), -1)  # 红点
            cv2.circle(img_cv, (x2, y2), 6, (0, 255, 0), -1)  # 绿点
        else:
            print(f"{path} → 未检测到端点")
            # 原图保存，无标记

        cv2.imwrite(save_path, img_cv)  # 不管是否检测成功都保存

