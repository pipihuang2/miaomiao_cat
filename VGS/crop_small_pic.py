import argparse
import os
import sys

import cv2
import numpy as np
import glob
import tqdm


def _execute(image, mask, fill_value) -> np.ndarray:
    """根据掩码提取 ROI 区域"""
    output_img = np.zeros_like(image)
    output_img[...] = fill_value

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    roi = mask > 0
    output_img[roi] = image[roi]

    return output_img


def main():
    parser = argparse.ArgumentParser(description="根据掩码提取 ROI 区域")
    parser.add_argument("input_dir", help="输入图片所在的目录路径")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"错误: 目录不存在 {input_dir}")
        sys.exit(1)

    # 查找掩码图片，优先查找目录下的 mask.png / mask.jpg
    mask_path = None
    for name in ("mask.png", "mask.jpg", "mask.bmp"):
        candidate = os.path.join(input_dir, name)
        if os.path.isfile(candidate):
            mask_path = candidate
            break

    if mask_path is None:
        print(f"错误: 在 {input_dir} 下未找到掩码图片 (mask.png / mask.jpg / mask.bmp)")
        sys.exit(1)

    mask_pic = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_pic is None:
        print(f"错误: 无法读取掩码图片 {mask_path}")
        sys.exit(1)

    # 创建输出目录
    output_dir = os.path.join(input_dir, "mask_output")
    os.makedirs(output_dir, exist_ok=True)

    # 递归查找所有 original_image.png
    pattern = os.path.join(input_dir, "**", "*original_image.png")
    pic_list = glob.glob(pattern, recursive=True)

    if not pic_list:
        print(f"警告: 在 {input_dir} 下未找到 *original_image.png 图片")
        sys.exit(0)

    for index, pic_path in tqdm.tqdm(enumerate(pic_list), total=len(pic_list)):
        image = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        output_img = _execute(image, mask_pic, 255)

        # 保持原文件名，前面加序号防止重名
        base_name = os.path.basename(pic_path)
        name, ext = os.path.splitext(base_name)
        # 如果原文件名都是 original_image.png，用上级目录名区分
        if name == "original_image":
            parent_name = os.path.basename(os.path.dirname(pic_path))
            out_name = f"{index:04d}_{parent_name}{ext}"
        else:
            out_name = f"{index:04d}_{base_name}"

        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, output_img)

    print(f"完成，共处理 {len(pic_list)} 张图片，结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
