import cv2
import numpy as np


def correct_skew(binary_img):
    # 使用 Canny 边缘检测
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)

    # 霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=200)

    # 可视化线条检测结果
    if lines is not None:
        line_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('detected_lines.png', line_img)

    if lines is None:
        print("未检测到直线，无法矫正倾斜")
        return binary_img, 0.0

    # 平均角度计算
    angles = []
    valid_lines = []  # 保存有效的线条参数
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi / 2) * (180 / np.pi)  # 转换为度数
        if abs(angle) < 8:  # 只考虑接近水平的直线
            angles.append(angle)
            valid_lines.append((rho, theta))

    # print('=== Detected line angles ===', angles)

    if not angles:
        print("未检测到直线，无法矫正倾斜")
        return binary_img, 0.0

    # 可视化保留的线条
    if valid_lines:
        valid_line_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for rho, theta in valid_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(valid_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色线条
        cv2.imwrite('valid_lines.png', valid_line_img)
        print(f"保存有效线条可视化图像: valid_lines.png (共{len(valid_lines)}条线)")

    # 计算平均角度
    average_angle = np.mean(angles)
    print(f"检测到的倾斜角度: {average_angle:.2f}°")

    # 旋转图像以矫正倾斜
    rotated_img= rotate_image(binary_img, average_angle)
    return rotated_img, average_angle

def crop_center_quarter(img):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 四分之一个长宽
    quarter_w = w // 4
    quarter_h = h // 4

    # 计算截取区域 (中心 ± 四分之一)
    x1 = cx - quarter_w
    x2 = cx + quarter_w
    y1 = cy - quarter_h
    y2 = cy + quarter_h

    # 裁剪
    cropped = img[y1:y2, x1:x2]
    return cropped

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵，scale=1 保持不缩放
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行仿射变换（保持原图大小）
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))  #
    return rotated

image = cv2.imread(r"E:\hyy\miaomiao_cat\F2_Cal\T1\1028\CAB_SF_20251028_105241175_1_right.bmp")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
center_img, angle, _, _, _ = correct_skew(crop_center_quarter(gray_image))
binary_rotate= rotate_image(image, angle)
cv2.imwrite(rf"E:\hyy\miaomiao_cat\F2_Cal\T1\1028\cal_pic\3_{angle}.png", binary_rotate)
