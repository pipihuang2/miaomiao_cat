import argparse
import os
import shutil
import json
import glob
import sys
import numpy as np
from PIL import Image, ImageDraw


def rasterize_polygon_mask(polygons_list, height, width, save_path=None):
    """
    将多边形列表栅格化为二值掩码。

    Args:
        polygons_list: [[(x1, y1), (x2, y2), ...], ...]
        height: 图像高度
        width: 图像宽度
        save_path: 若提供，则保存为黑白 PNG

    Returns:
        HxW bool 掩码（True=前景，False=背景）
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for polygon in polygons_list:
        if polygon is None or len(polygon) < 3:
            continue
        xy = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(xy, outline=255, fill=255)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        mask.save(save_path)
        print(f"✅ Saved mask to {save_path}")

    return np.array(mask, dtype=bool)


def main():
    parser = argparse.ArgumentParser(description="从 Labelme JSON 中提取 glue 标注并生成掩码")
    parser.add_argument("input_dir", help="输入目录路径（包含 .json 和对应图片）")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"❌ 错误: 目录不存在 {input_dir}")
        sys.exit(1)

    # 自动创建输出目录
    img_out_dir = os.path.join(input_dir, "Image")
    mask_out_dir = os.path.join(input_dir, "Instance")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    json_list = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_list:
        print(f"⚠️ 警告: 在 {input_dir} 下未找到 .json 文件")
        sys.exit(0)

    for json_path in json_list:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取 label == "glue" 的多边形
        glue_polygons = []
        for shape in data.get("shapes", []):
            if shape.get("label") == "glue":
                points = shape.get("points", [])
                if len(points) >= 3:
                    glue_polygons.append(points)

        if len(glue_polygons) == 0:
            print(f"⏭️ 跳过，没有 glue 标注: {json_path}")
            continue

        # 查找对应的原图
        image_path = data.get("imagePath")
        if not image_path:
            print(f"⚠️ JSON 中没有 imagePath: {json_path}")
            continue

        original_pic = os.path.join(input_dir, image_path)
        if not os.path.exists(original_pic):
            # 尝试直接用 JSON 同目录下的图片
            fallback_pic = os.path.join(input_dir, os.path.basename(image_path))
            if os.path.exists(fallback_pic):
                original_pic = fallback_pic
            else:
                print(f"❌ 没有找到对应图片: {image_path}")
                continue

        # 复制原图到 Image 目录
        dst_img_path = os.path.join(img_out_dir, os.path.basename(image_path))
        shutil.copy2(original_pic, dst_img_path)

        # 生成并保存掩码到 Instance 目录
        mask_name = os.path.basename(json_path).replace(".json", ".png")
        out_mask_path = os.path.join(mask_out_dir, mask_name)

        rasterize_polygon_mask(
            glue_polygons,
            data["imageHeight"],
            data["imageWidth"],
            out_mask_path
        )

    print(f"\n🎉 完成！输出目录：\n  原图: {img_out_dir}\n  掩码: {mask_out_dir}")


if __name__ == "__main__":
    main()
