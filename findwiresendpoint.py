import cv2
import numpy as np
from pathlib import Path

def detect_insertions(
        img_path: str | Path,
        *,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
        hough_thresh: int = 50,
        min_line_len: int = 50,
        max_line_gap: int = 15,
        merge_dist_px: int = 8,
        dot_radius: int = 4,
        bgr_dot: tuple[int,int,int] = (0, 0, 255)   # 红色 (B,G,R)
    ):
    """
    检测线缆插入基板的位置。

    返回
    ----
    coords : list[(x,y)] ── 以图片左下为 (0,0) 的坐标
    annot  : ndarray      ── 已标红点的 BGR 图像
    """
    img_path = Path(img_path)
    img     = cv2.imread(str(img_path))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 边缘检测
    edges   = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    # 2. 直线检测
    lines   = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return [], img.copy()

    h, w = gray.shape
    endpts = []

    # 3. 过滤出“线缆”──基本是斜向、长度足够
    for x1,y1,x2,y2 in lines[:,0]:
        # 去除接近水平/垂直的杂线
        dx, dy = x2 - x1, y2 - y1
        angle  = np.degrees(np.arctan2(dy, dx))
        if 15 < abs(angle) < 75:          # 粗调：倾斜 15°–75°
            # 记录更靠近底部的端点（y 值较大）
            if y1 > y2:
                endpts.append((x1, y1))
            else:
                endpts.append((x2, y2))

    if not endpts:
        return [], img.copy()

    # 4. 聚类 / 合并距离很近的端点，避免同一插口被重复检测
    endpts = np.array(endpts, dtype=np.float32)
    merged = []
    taken  = np.zeros(len(endpts), bool)

    for i, p in enumerate(endpts):
        if taken[i]:
            continue
        group = [p]
        taken[i] = True
        # 将欧氏距离 < merge_dist_px 的点并成一簇
        dists = np.linalg.norm(endpts - p, axis=1)
        close = np.where((dists < merge_dist_px) & (~taken))[0]
        for j in close:
            group.append(endpts[j])
            taken[j] = True
        merged.append(np.mean(group, axis=0))   # 取平均作为该簇代表

    merged = np.array(merged, dtype=int)

    # 5. 画红点 & 转换坐标系
    annot = img.copy()
    coords = []
    for x,y in merged:
        cv2.circle(annot, (int(x), int(y)), dot_radius, bgr_dot, -1)
        # OpenCV 原点在左上；题目要求左下 → y' = h - 1 - y
        coords.append((int(x), int(h - 1 - y)))

    # 若需要保存：
    # out = img_path.parent / f"{img_path.stem}_annot{img_path.suffix}"
    # cv2.imwrite(str(out), annot)

    return coords, annot


# ------------------- 批量处理示例 -------------------
if __name__ == "__main__":
    src_dir = Path("./samples")      # 放原始图片的位置
    dst_dir = Path("./annotated")    # 存结果
    dst_dir.mkdir(exist_ok=True)

    for jpg in src_dir.glob("*.jpg"):
        pts, ann = detect_insertions(jpg)
        print(f"{jpg.name}: {pts}")

        # 保存标注图
        cv2.imwrite(str(dst_dir / jpg.name), ann)