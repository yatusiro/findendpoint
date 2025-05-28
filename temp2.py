# processpoint.py
import numpy as np

class ProcessPoint:
    """
    • 将左上角为原点的像素坐标 → 以图像中心为原点的毫米坐标  
    • 判断是否落在 ±limit_mm 范围  
    """

    def __init__(
        self,
        px_per_mm: float = 2.0,               # “多少像素 = 1 mm”
        limit_mm: float = 10.0,               # 判定阈值
        matrix: np.ndarray | None = None,     # 2×2 或 3×3 校正矩阵
        y_axis_up: bool = True,               # True: y 向上为正
    ):
        self.px_per_mm = px_per_mm
        self.limit_mm = limit_mm
        self.y_axis_up = y_axis_up

        # 如果没给，就用单位矩阵
        if matrix is None:
            self.matrix = np.eye(2)           # [[1,0],[0,1]]
        else:
            self.matrix = np.asarray(matrix, dtype=float)

    # ── 像素 → 毫米 (中心原点) ───────────────────────────────
    def pixel_to_mm_center(
        self, point_px: tuple[int, int], W: int, H: int
    ) -> tuple[float, float]:
        cx, cy = W / 2, H / 2
        x_c = point_px[0] - cx
        y_c = cy - point_px[1] if self.y_axis_up else point_px[1] - cy

        vec_px = np.array([x_c, y_c], dtype=float)
        vec_mm = vec_px / self.px_per_mm          # 像素 → mm
        vec_mm_corr = self.matrix @ vec_mm        # 校正

        return float(vec_mm_corr[0]), float(vec_mm_corr[1])

    # ── 范围判定 ────────────────────────────────────────────
    def evaluate(
        self, point_px: tuple[int, int], W: int, H: int
    ) -> tuple[bool, tuple[float, float], str]:
        """
        返回 (是否在范围内, 毫米坐标 (x_mm, y_mm), 建议颜色 'green'/'red')
        """
        x_mm, y_mm = self.pixel_to_mm_center(point_px, W, H)
        ok = abs(x_mm) <= self.limit_mm and abs(y_mm) <= self.limit_mm
        color = "green" if ok else "red"
        return ok, (x_mm, y_mm), color
