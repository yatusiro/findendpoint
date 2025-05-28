class ProcessPoint:
    def __init__(
        self,
        matrix_path: str | None = "config/matrix.toml",
        px_per_mm: float = 2.0,       # ← 加回：外部可改
        limit_mm: float = 10.0,
        y_axis_up: bool = True,
    ):
        self.px_per_mm = px_per_mm
        self.limit_mm = limit_mm
        self.y_axis_up = y_axis_up
        self.matrix = (
            self._load_matrix(matrix_path)
            if matrix_path is not None
            else np.eye(2)
        )

    # 像素 → 毫米：先除 px_per_mm，再乘矩阵
    def pixel_to_mm_center(self, point_px, W, H):
        cx, cy = W / 2, H / 2
        x_c = point_px[0] - cx
        y_c = cy - point_px[1] if self.y_axis_up else point_px[1] - cy

        vec_px = np.array([x_c, y_c], dtype=float)
        vec_mm = vec_px / self.px_per_mm          # ← 缩放
        vec_mm_corr = self.matrix @ vec_mm        # ← 校正
        return float(vec_mm_corr[0]), float(vec_mm_corr[1])
