# processpoint.py
from pathlib import Path
import numpy as np
import tomllib            # Python 3.11 以上；若用 3.10 → pip install tomli

class ProcessPoint:
    """
    将“左上角为原点”的像素坐标 → “图像中心为原点”的毫米坐标，
    并判断是否落在 ±limit_mm 的范围内。
    """

    def __init__(
        self,
        matrix_path: str = "config/matrix.toml",
        px_per_mm: float = 2.0,       # 2 像素 = 1 mm
        limit_mm: float = 10.0,       # 判定阈值
        y_axis_up: bool = True,       # True: y 向上为正；False: 向下为正
    ):
        self.px_per_mm = px_per_mm
        self.limit_mm = limit_mm
        self.y_axis_up = y_axis_up
        self.matrix = self._load_matrix(matrix_path)

    # ---------- 私有方法 --------------------------------------------------
    @staticmethod
    def _load_matrix(path: str) -> np.ndarray:
        """
        读取 config/matrix.toml，返回 2×2 或 3×3 校正矩阵
        toml 示例:
        [matrix]
        data = [[1,0],[0,1]]
        """
        data = tomllib.loads(Path(path).read_text())
        return np.asarray(data["matrix"]["data"], dtype=float)

    # ---------- 公共 API --------------------------------------------------
    def pixel_to_mm_center(
        self, point_px: tuple[int, int], W: int, H: int
    ) -> tuple[float, float]:
        """
        (x_px, y_px)  →  (x_mm, y_mm)，新坐标系以图像中心为原点。
        """
        cx, cy = W / 2, H / 2
        x_c = point_px[0] - cx
        y_c = cy - point_px[1] if self.y_axis_up else point_px[1] - cy

        vec_px = np.array([x_c, y_c], dtype=float)
        vec_mm = vec_px / self.px_per_mm            # 像素 → mm
        vec_mm_corr = self.matrix @ vec_mm          # 校正

        return float(vec_mm_corr[0]), float(vec_mm_corr[1])

    def evaluate(
        self, point_px: tuple[int, int], W: int, H: int
    ) -> tuple[bool, tuple[float, float], str]:
        """
        1) 把像素坐标转到毫米坐标
        2) 判断 |x|, |y| 是否 ≤ limit_mm
        返回: (是否在范围内, 毫米坐标, 建议颜色字符串 'green'/'red')
        """
        x_mm, y_mm = self.pixel_to_mm_center(point_px, W, H)
        ok = abs(x_mm) <= self.limit_mm and abs(y_mm) <= self.limit_mm
        color = "green" if ok else "red"
        return ok, (x_mm, y_mm), color









[matrix]
data = [
  [1.0, 0.0],
  [0.0, 1.0]
]

