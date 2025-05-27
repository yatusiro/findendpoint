def preprocess_image(self, image: Image.Image) -> Image.Image:
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 
                                   11,  # 小さいほど細かく検出出来る（奇数）
                                   2   # 大きほと白になりにくい
                                   )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H = cv_image.shape[0]
    top_threshold = int(H * 0.2)
    bottom_threshold = int(H * 0.8)

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            # 获取轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            center_y = y + h // 2

            # 只处理中间区域的轮廓（排除顶部20%和底部20%）
            if top_threshold <= center_y <= bottom_threshold:
                cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
