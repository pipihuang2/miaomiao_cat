import os
import shutil

import cv2
import json
import glob
import numpy as np
from PIL import Image, ImageDraw

from PIL import Image, ImageDraw
import numpy as np
import os

def rasterize_polygon_mask(polygons, height, width, save_path=None):
    """
    polygons: List[List[float]] 或 [(x,y),...]
    返回: HxW bool 掩码（True=前景，False=背景）
    如果提供 save_path，则同时保存为黑白PNG图像。
    """
    # 创建黑色背景的二值图
    mask = Image.new("L", (width, height), 0)  # "L"=灰度模式
    draw = ImageDraw.Draw(mask)

    # 确保坐标格式正确
    xy = [(float(x), float(y)) for x, y in polygons]

    # 画出白色区域（目标）
    draw.polygon(xy, outline=255, fill=255)

    # 保存文件（可选）
    if save_path:
        # 自动创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask.save(save_path)
        print(f"✅ Saved mask to {save_path}")

    # 返回 numpy 掩码
    return np.array(mask, dtype=bool)


json_list = glob.glob(r"D:\Project\HYJ_Pic\new_cab\20260104\CamA\20260104\have\*.json")
for json_path in json_list:
    with open(json_path, "r") as f:
        data = json.load(f)
        if len(data["shapes"]) == 1 and data["shapes"][0].get("label") == "glue":
            original_pic_png = os.path.join(r"D:\Project\HYJ_Pic\new_cab\20260104\CamA\20260104\have",data.get("imagePath"))
            if not os.path.exists(original_pic_png):
                print(f'没有找到{original_pic_png}文件')
                continue
            shutil.copy(original_pic_png, os.path.join(r"D:\Project\HYJ_Pic\new_cab\20260104\CamA\20260104\Image",data.get("imagePath")))
            polygons_ = data.get('shapes', [])[0].get('points')
            out_pic_name = os.path.join(r"D:\Project\HYJ_Pic\new_cab\20260104\CamA\20260104\Instance", os.path.basename(json_path).replace(".json", ".png"))
            pic_image = rasterize_polygon_mask(polygons_, data["imageHeight"], data["imageWidth"],out_pic_name)
