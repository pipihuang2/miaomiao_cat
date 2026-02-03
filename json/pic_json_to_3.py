import os
import json
import copy
import cv2
from tqdm import tqdm   # 新增：进度条

def split_image_and_json(json_path, img_root_dir, out_dir,
                         min_box_w=5, min_box_h=5,
                         split_json=True):
    """
    split_json: 是否需要分割 json
                True = 分割图片 & json（默认）
                False = 只分割图片，不输出 json
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    if split_json:
        os.makedirs(os.path.join(out_dir, "jsons"), exist_ok=True)

    # 读取 json（如果不分割 json，只需要读取 img_path）
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_name = data["imagePath"]
    img_path = os.path.join(img_root_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图片读不到: {img_path}")

    h, w = img.shape[:2]

    # 三段分割
    w1 = w // 3
    w2 = w // 3
    w3 = w - w1 - w2

    slice_ranges = [
        (0, w1),
        (w1, w1 + w2),
        (w1 + w2, w)
    ]

    base_name, ext = os.path.splitext(img_name)

    for i, (x_start, x_end) in enumerate(slice_ranges):

        # -------------- 只分割图片 --------------
        sub_img = img[:, x_start:x_end]
        sub_img_name = f"{base_name}_part{i}{ext}"
        sub_img_path = os.path.join(out_dir, "images", sub_img_name)
        cv2.imwrite(sub_img_path, sub_img)
        # ---------------------------------------

        if not split_json:
            continue  # 如果不需要 json，直接跳过后续处理

        # ================= 分割 json 的原始逻辑 =================
        import copy
        sub_data = copy.deepcopy(data)
        sub_data["imagePath"] = sub_img_name
        sub_data["imageWidth"] = x_end - x_start
        sub_data["imageHeight"] = h

        new_shapes = []

        for shape in data["shapes"]:
            pts = shape["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            inter_x1 = max(x1, x_start)
            inter_x2 = min(x2, x_end)
            if inter_x2 <= inter_x1:
                continue

            inter_y1 = max(y1, 0)
            inter_y2 = min(y2, h)
            if inter_y2 <= inter_y1:
                continue

            box_w = inter_x2 - inter_x1
            box_h = inter_y2 - inter_y1

            if box_w < min_box_w or box_h < min_box_h:
                continue

            new_pts = [
                [inter_x1 - x_start, inter_y1],
                [inter_x2 - x_start, inter_y1],
                [inter_x2 - x_start, inter_y2],
                [inter_x1 - x_start, inter_y2],
            ]
            new_shape = copy.deepcopy(shape)
            new_shape["points"] = new_pts
            new_shapes.append(new_shape)

        sub_data["shapes"] = new_shapes

        sub_json_name = f"{base_name}_part{i}.json"
        sub_json_path = os.path.join(out_dir, "jsons", sub_json_name)
        with open(sub_json_path, "w", encoding="utf-8") as f:
            json.dump(sub_data, f, ensure_ascii=False, indent=2)

    print(f"完成: {json_path}")



if __name__ == "__main__":
    import glob

    json_dir = r"D:\Project\HYJ_Pic\F2\1122\true_have"
    img_root_dir = r"D:\Project\HYJ_Pic\F2\1122\true_have"
    out_dir = r"D:\Project\HYJ_Pic\F2\1122\true_have\3"

    # 收集所有 json
    json_list = glob.glob(os.path.join(json_dir, "*.json"))

    # 用 tqdm 显示进度条
    for jp in tqdm(json_list, desc="Processing JSON files"):
        # 这里可以按需修改最小框尺寸
        split_image_and_json(jp, img_root_dir, out_dir,
                             min_box_w=50,   # 最小宽度阈值
                             min_box_h=50,
                             split_json = False)   # 最小高度阈值
