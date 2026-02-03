import json
import cv2
import os
from copy import deepcopy

# ========================
# 配置区：改成你自己的路径
# ========================
input_dir = r"D:\Project\HYJ_Pic\F2\true_have\1\1111"   # 存放原始 json 和图片的文件夹
output_dir = r"D:\Project\HYJ_Pic\F2\true_have\1\crops\1" # 想要保存裁剪结果（图片+json）的文件夹
target_size = 1500                 # 目标正方形边长：1000 像素
# ========================


os.makedirs(output_dir, exist_ok=True)


def get_shape_center(shape):
    """根据 LabelMe 的 points 求中心（适用于矩形、多边形）"""
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return cx, cy


def crop_max_square(img, cx, cy, target_size=1000):
    """
    以 (cx, cy) 为中心裁剪一个正方形区域：
    - 尽量为 target_size × target_size
    - 若受边界限制，则取不越界前提下的最大正方形
    返回：crop_img, (x1, y1, x2, y2)
    """
    H, W = img.shape[:2]
    half = target_size / 2.0

    max_left  = cx
    max_right = W - 1 - cx
    max_top   = cy
    max_bot   = H - 1 - cy

    half_actual = min(half, max_left, max_right, max_top, max_bot)
    half_actual = max(1, int(half_actual))

    x1 = int(round(cx - half_actual))
    x2 = int(round(cx + half_actual))
    y1 = int(round(cy - half_actual))
    y2 = int(round(cy + half_actual))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def process_one_json(json_path):
    """处理单个 json + 对应图片"""
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    img_name = ann["imagePath"]
    # 图片和 json 在同一个文件夹
    img_path = os.path.join(os.path.dirname(json_path), img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读图失败，跳过: {img_path}")
        return

    H, W = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_name))[0]

    for idx, shape in enumerate(ann.get("shapes", [])):
        cx, cy = get_shape_center(shape)
        crop_img, (x1, y1, x2, y2) = crop_max_square(img, cx, cy, target_size=target_size)
        ch, cw = crop_img.shape[:2]

        print(
            f"{os.path.basename(json_path)} | shape {idx}: "
            f"center=({cx:.1f},{cy:.1f}), crop=({x1},{y1})-({x2},{y2}), size={cw}x{ch}"
        )

        # ---- 保存裁剪后的图片 ----
        crop_img_name = f"{base_name}_shape{idx}_crop.jpg"
        crop_img_path = os.path.join(output_dir, crop_img_name)
        cv2.imwrite(crop_img_path, crop_img)

        # ---- 生成对应的 JSON ----
        new_ann = {
            "version": ann.get("version", "3.0.0"),
            "flags": deepcopy(ann.get("flags", {})),
            "shapes": [],
            "imagePath": crop_img_name,
            "imageData": None,
            "imageHeight": ch,
            "imageWidth":  cw,
            "description": ann.get("description", ""),
        }

        # 只保留当前这个 shape，坐标减去 (x1, y1)
        new_shape = deepcopy(shape)
        new_points = []
        for (px, py) in shape["points"]:
            new_points.append([px - x1, py - y1])
        new_shape["points"] = new_points
        new_ann["shapes"].append(new_shape)

        crop_json_name = f"{base_name}_shape{idx}_crop.json"
        crop_json_path = os.path.join(output_dir, crop_json_name)
        with open(crop_json_path, "w", encoding="utf-8") as f:
            json.dump(new_ann, f, ensure_ascii=False, indent=2)


def main():
    # 遍历 input_dir 下所有 .json
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue
        json_path = os.path.join(input_dir, fname)
        try:
            process_one_json(json_path)
        except Exception as e:
            print(f"[ERROR] 处理 {json_path} 出错: {e}")


if __name__ == "__main__":
    main()
