# processpoint.py  ── 标尺/旋转/剪切全交给矩阵 ──────────────
import numpy as np
from pathlib import Path
import tomllib          # 3.11+；若 3.10 → pip install tomli


class ProcessPoint:
    """
    把左上角为原点的像素坐标 → 图像中心为原点的毫米坐标，
    所有缩放 / 旋转 / 剪切都在矩阵里完成。
    """

    def __init__(
        self,
        matrix_path: str | None = "config/matrix.toml",
        limit_mm: float = 10.0,       # 判定阈值
        y_axis_up: bool = True        # True: y 向上为正；False: 向下为正
    ):
        self.limit_mm = limit_mm
        self.y_axis_up = y_axis_up

        # 若未给路径则用单位矩阵
        self.matrix = (
            self._load_matrix(matrix_path)
            if matrix_path is not None
            else np.eye(2)
        )

    # ── 私有：读 TOML 中的矩阵 ------------------------------------------
    @staticmethod
    def _load_matrix(path: str) -> np.ndarray:
        """
        读取 [matrix].data (2×2 或 3×3)，返回 np.ndarray。
        若需 3×3，请在 pixel_to_mm_center 里自行扩展/除以齐次坐标。
        """
        raw = tomllib.loads(Path(path).read_text())
        return np.asarray(raw["matrix"]["data"], dtype=float)

    # ── 像素 → 毫米（中心原点） ------------------------------------------
    def pixel_to_mm_center(
        self, point_px: tuple[int, int], W: int, H: int
    ) -> tuple[float, float]:
        """
        (x_px, y_px) ➜ (x_mm, y_mm)，全靠 self.matrix 校正。
        self.matrix 必须包含实际缩放因子：
        例：2 px = 1 mm →  0.5 缩放写进矩阵对角线。
        """
        cx, cy = W / 2, H / 2
        x_c = point_px[0] - cx
        y_c = cy - point_px[1] if self.y_axis_up else point_px[1] - cy

        vec_px = np.array([x_c, y_c], dtype=float)         # 2×1 向量
        vec_mm = self.matrix @ vec_px                      # 线性校正
        return float(vec_mm[0]), float(vec_mm[1])

    # ── 判定范围 ----------------------------------------------------------
    def evaluate(
        self, point_px: tuple[int, int], W: int, H: int
    ) -> tuple[bool, tuple[float, float], str]:
        """
        返回: (是否在 ±limit_mm 内, 毫米坐标 (x, y), 建议颜色 'green'/'red')
        """
        x_mm, y_mm = self.pixel_to_mm_center(point_px, W, H)
        ok = abs(x_mm) <= self.limit_mm and abs(y_mm) <= self.limit_mm
        return ok, (x_mm, y_mm), ("green" if ok else "red")
