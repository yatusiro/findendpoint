import cv2
import os
import glob
import numpy as np
from typing import List, Tuple
from PIL import Image
from math import atan2, pi
import toml

class WireEndpointDetector:
    def __init__(self, config_path="classifier_config.toml"):
        # 从配置文件读取参数
        config = toml.load(config_path)

        self.hough_thresh = config["hough_thresh"]
        self.min_len = config["min_len"]
        self.max_gap = config["max_gap"]
        self.tol_theta = config["tol_theta_deg"] * pi / 180
        self.tol_rho_px = config["tol_rho_px"]
        self.dot_radius = config["dot_radius"]
        self.angle_thr_deg = config["angle_thr_deg"]
        self.edge_pct = config["edge_pct"]
        self.side_pct = config["side_pct"]

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

    def detect_endpoints(self, img_pil: Image.Image) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
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
            return img_pil.copy(), []

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

        endpoints, keep_pairs = [], []
        top_band = self.edge_pct * H
        bottom_band = (1 - self.edge_pct) * H
        left_thr = self.side_pct * W
        right_thr = (1 - self.side_pct) * W

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

            endpoints.append((x1 - cx, cy - y1, x2 - cx, cy - y2))
            keep_pairs.append(((x1, y1), (x2, y2)))

        vis = bgr.copy()
        for (p1, p2) in keep_pairs:
            for (x, y) in (p1, p2):
                cv2.circle(vis, (int(x), int(y)), self.dot_radius, (255, 0, 0), -1)

        return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), endpoints

    def run_on_folder(self, folder_path: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        for path in jpg_files:
            print(f"Processing {path}")
            img = Image.open(path)
            processed_img = self.preprocess_image(img)
            annotated, _ = self.detect_endpoints(processed_img)
            name, ext = os.path.splitext(os.path.basename(path))
            output_path = os.path.join(output_folder, f"{name}_processed{ext}")
            annotated.save(output_path)


