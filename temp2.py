if __name__ == "__main__":
    os.makedirs("output2", exist_ok=True)
    os.makedirs("output3", exist_ok=True)
    detector = ProbabilisticHough()

    image_paths = glob.glob("input1/*.jpg")  
    for path in image_paths:
        image = Image.open(path)
        
        endpoints = detector.detect_points(image)
        processed_image = detector.preprocess_image(image)

        img_bgr = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        H, W = img_bgr.shape[:2]
        cx, cy = W // 2, H // 2

        if endpoints:
            print(f"{path} → 检测到 {len(endpoints)} 组端点")
            for idx, (pt1, pt2) in enumerate(endpoints):
                print(f"  组 {idx + 1}: {pt1}, {pt2}")
                x1, y1 = int(cx + pt1[0]), int(cy - pt1[1])
                x2, y2 = int(cx + pt2[0]), int(cy - pt2[1])
                cv2.circle(img_bgr, (x1, y1), detector.dot_radius, (255, 0, 0), -1)
                cv2.circle(img_bgr, (x2, y2), detector.dot_radius, (0, 255, 0), -1)

        merged = cv2.hconcat([
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
            img_bgr  # 已在 processed_image 上绘制端点
        ])        
        save_path = os.path.join("output6", os.path.basename(path))
        cv2.imwrite(save_path, merged)
