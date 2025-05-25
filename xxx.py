from typing import List, Tuple
from math import atan2, pi
import cv2, numpy as np
from PIL import Image

def detect_red_line_points(
    img_pil: Image.Image,
    *,
    hough_thresh: int = 30,
    min_len: int = 30,
    max_gap: int = 7,
    tol_theta_deg: float = 8.0,
    tol_rho_px: float = 25.0,
    margin_px: int | None = 5,   # ← None 表示不过滤
    dot_radius: int = 6,
    draw: bool = True,
) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    """
    Detect red-line endpoints.  Returns (annotated_image, flat_point_list).
    flat_point_list = [(x, y), ...] in centre-origin coords.
    """

    # 0. PIL → BGR
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # 1. 红色掩膜
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0,100,80), (10,255,255))
    mask |= cv2.inRange(hsv, (160,100,80), (180,255,255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    # 2. Hough
    segs = cv2.HoughLinesP(
        mask, 1, pi/180,
        threshold=hough_thresh,
        minLineLength=min_len,
        maxLineGap=max_gap)
    if segs is None:
        return img_pil.copy() if draw else img_pil, []

    segs = segs[:,0]                        # (N,4)

    # 3. θρ 聚类
    tol_theta = tol_theta_deg * pi/180
    groups: list[list] = []                # [θ̄, ρ̄, [segs]]
    for x1,y1,x2,y2 in segs:
        θ = atan2(y2-y1, x2-x1)
        if θ < 0: θ += pi
        ρ = np.sin(θ)*x1 - np.cos(θ)*y1
        for g in groups:
            θm, ρm, s = g
            if min(abs(θ-θm), pi-abs(θ-θm)) < tol_theta and abs(ρ-ρm) < tol_rho_px:
                s.append((x1,y1,x2,y2))
                k = len(s)
                g[0] = (θm*(k-1)+θ)/k
                g[1] = (ρm*(k-1)+ρ)/k
                break
        else:
            groups.append([θ, ρ, [(x1,y1,x2,y2)]])

    # 4. 每组选最远两点 → points_flat
    points: list[tuple[float,float]] = []
    for _,_,gsegs in groups:
        pts = np.array([(x1,y1) for x1,y1,_,_ in gsegs] +
                       [(x2,y2) for _,_,x2,y2 in gsegs])
        if len(pts) < 2: continue
        d2 = np.sum((pts[:,None]-pts[None])**2, axis=2)
        i,j = np.unravel_index(np.argmax(d2), d2.shape)
        for idx in (i,j):
            x,y = pts[idx]
            # 5. 贴边判定（可选）
            if margin_px is not None:
                px,py = x, y
                if (px < margin_px or px > W-margin_px or
                    py < margin_px or py > H-margin_px):
                    continue
            points.append((x-cx, cy-y))     # 转中心原点

    # 6. 绘制蓝点
    if not draw:
        return img_pil, points

    vis = bgr.copy()
    blue = (255,0,0)                        # BGR
    for dx,dy in points:
        cv2.circle(vis, (int(dx+cx), int(cy-dy)), dot_radius, blue, -1)

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, points


# ───────── quick demo ─────────
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open("test.jpg")
    annot, pts = detect_red_line_points(img)

    print("端点数量:", len(pts))
    for i,(x,y) in enumerate(pts,1):
        print(f"{i:02d}: ({x:+.1f}, {y:+.1f})")

    plt.imshow(annot); plt.axis("off"); plt.show()
