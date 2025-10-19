"""
将图像切割为多个小块，并保存到指定的输出文件夹中。

输入：
- 输入文件夹：包含需要切割的图像文件
- 输出文件夹：保存切割后的小块图像
- 切片大小：每个小块的尺寸
- 预处理：是否对图像进行预处理（二值化+裁剪）
- 调试模式：是否输出调试信息
"""
import glob

from PIL import Image
import os
from tqdm import tqdm  # 导入 tqdm
import math
import cv2
import numpy as np

def preprocess_image(img_path, output_dir=None, debug=False):
    """
    对图像进行预处理：先读取图像、旋转、二值化、提取最大轮廓，
    得到 cropped_output（仅保留最大轮廓内区域）。
    """
    # 1. 读取图像（使用 cv2）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("图像读取失败，请检查文件路径！")
    # 此处根据需要进行旋转（例如逆时针90度）
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. 二值化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, binary_global = cv2.threshold(gray_blurred, 100, 255, cv2.THRESH_BINARY)
    if debug and output_dir is not None:
        cv2.imwrite(os.path.join(output_dir, "binary_global.png"), binary_global)

    # 3. 获取全局轮廓，并找出最大轮廓
    contours, hierarchy = cv2.findContours(binary_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    if largest_contour is not None:
        # 创建掩膜，并用最大轮廓填充
        mask = np.zeros_like(binary_global)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)
        # 构建白色背景图
        output_img = np.full_like(img, 255)
        output_img[mask == 255] = img[mask == 255]
        # 获取最大轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_output = output_img[y:y+h, x:x+w]
        return cropped_output
    else:
        # 若没有找到有效轮廓，则返回原图
        return img

def slice_single_image(index, input_image_path, output_folder_for_slices, slice_size=(512, 512), preprocess=False, debug=False):
    """
    【辅助函数】
    切割单张图像为多个小块并保存到指定的单一输出文件夹。
    如果 preprocess 为 True，则先对输入图片进行预处理（包括二值化及裁剪），
    得到 cropped_output 后再进行切割处理。
    """
    try:
        # 若选择预处理，则先调用 preprocess_image 得到裁剪后的图像
        if preprocess:
            cropped_img = preprocess_image(input_image_path, output_dir=output_folder_for_slices, debug=debug)
            # 将 OpenCV 格式图像（默认 BGR）转换为 PIL 格式（RGB）
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cropped_img)
        else:
            img = Image.open(input_image_path)

        img_width, img_height = img.size
        image_name = os.path.splitext(os.path.basename(input_image_path))[0]
        print(f"  - 正在切割图像: {os.path.basename(input_image_path)} ({img_width}x{img_height})")

        # 计算切片总数（向上取整）
        num_slices_x = math.ceil(img_width / slice_size[0])
        num_slices_y = math.ceil(img_height / slice_size[1])
        total_slices = num_slices_x * num_slices_y

        slice_width, slice_height = slice_size
        slice_count = 0

        with tqdm(total=total_slices, desc=f"    切片 {image_name}", unit="切片", leave=False) as pbar:
            for i in range(0, img_height, slice_height):
                for j in range(0, img_width, slice_width):
                    box = (j, i, min(j + slice_width, img_width), min(i + slice_height, img_height))
                    sliced_img = img.crop(box)

                    # 将PIL Image转换为numpy数组
                    sliced_img_array = np.array(sliced_img)

                    # 直接分离RGB通道（PIL Image是RGB格式）
                    r, g, b = sliced_img_array[:, :, 0], sliced_img_array[:, :, 1], sliced_img_array[:, :, 2]
                    gray = (0.15 * r + 0.15 * g + 0.7 * b).astype(np.uint8)
                    gray = cv2.blur(gray, (25, 25))

                    # 边缘检测
                    kernel_x = np.array([30, 5, 0, -5, -30]).reshape(1, 5)  # 水平方向
                    kernel_y = np.array([30, 5, 0, -5, -30]).reshape(5, 1)  # 垂直方向
                    # kernel_x = np.array([20, 10, 1, 0, -1, -10, -20]).reshape(1, 7)
                    # kernel_y = np.array([20, 10, 1, 0, -1, -10, -20]).reshape(7, 1)
                    edge_x = cv2.filter2D(gray, -1, kernel_x)
                    edge_y = cv2.filter2D(gray, -1, kernel_y)

                    if (np.any(np.abs(edge_x) > 60) or np.any(np.abs(edge_y) > 60)):
                        slice_filename = f"{index}_{image_name}_{slice_count}.png"
                        output_path = os.path.join(output_folder_for_slices, slice_filename)
                        sliced_img.save(output_path)
                        slice_count += 1
                    pbar.update(1)
    except FileNotFoundError:
        print(f"错误: 图像文件未找到 - {input_image_path}")
    except Exception as e:
        print(f"处理图像 {os.path.basename(input_image_path)} 时发生错误: {e}")

def slice_images_in_folder(input_folder, output_folder_for_all_slices, slice_size=(512, 512), preprocess=False, debug=False):
    """
    遍历指定文件夹下的所有图片文件，
    对每张图片根据 preprocess 参数决定是否预处理后切割，
    将所有切片保存在同一个输出文件夹中。
    """
    print(f"--- 开始处理文件夹: {input_folder} ---")

    if not os.path.isdir(input_folder):
        print(f"错误: 输入文件夹未找到或不是一个目录 - {input_folder}")
        return

    if not os.path.exists(output_folder_for_all_slices):
        os.makedirs(output_folder_for_all_slices)
        print(f"创建单一输出文件夹: {output_folder_for_all_slices}")

    image_files = glob.glob(os.path.join(input_folder, "**","original_image.jpg"),recursive=True)

    for index, filename in tqdm(enumerate(image_files),total=len(image_files)):
        input_image_path = os.path.join(input_folder, filename)
        slice_single_image(index, input_image_path, output_folder_for_all_slices, slice_size, preprocess, debug)

    print(f"\n--- 所有图片处理完成 ---")

if __name__ == "__main__":
    input_folder = r"D:\Project\HYJ_Pic\vgs\20251011"
    base_output_folder = r"D:\Project\HYJ_Pic\vgs\2"
    slice_size = (256, 256)
    # 设置 preprocess=True 则先进行预处理（二值化+裁剪），否则直接切割原图
    slice_images_in_folder(input_folder, base_output_folder, slice_size, preprocess=True, debug=False)