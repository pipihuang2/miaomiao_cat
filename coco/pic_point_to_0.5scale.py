import os
import json
import cv2
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 设置路径
input_dir = r"E:\yrt\kong\data_0425\val"
output_dir = r"E:\yrt\kong\data_0425\resize_val"
os.makedirs(output_dir, exist_ok=True)

# 缩放因子
scale = 0.5

# 中文 → 英文标签映射（可选）
label_mapping = {
    "断胶": "ng",
    "残胶污迹": "ng",
    "异物混入": "ng",
    "接头缺胶": "ng",
    "胶分层": "ng",
    "记号笔印": "ng",
    "收胶不良": "ng",
    "胶宽超宽": "ng",
    "气泡": "ng",
    "透光": "ng",
    "胶宽不足": "ng",
    "cut assy": "ng",
}

# 每个任务处理函数
def process_file(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_name = data["imagePath"]
        img_path = os.path.join(input_dir, img_name)
        if not os.path.exists(img_path):
            return f"[跳过] 找不到图片: {img_path}"

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 保存缩放图像
        new_img_name = os.path.splitext(img_name)[0] + "_resized.jpg"
        new_img_path = os.path.join(output_dir, new_img_name)
        cv2.imwrite(new_img_path, resized_img)

        # 修改 JSON 内容
        data["imageWidth"] = int(w * scale)
        data["imageHeight"] = int(h * scale)
        data["imagePath"] = new_img_name

        for shape in data["shapes"]:
            if shape["label"] in label_mapping:
                shape["label"] = label_mapping[shape["label"]]
            shape["points"] = [[x * scale, y * scale] for x, y in shape["points"]]

        # 保存新 JSON 文件
        new_json_name = os.path.splitext(os.path.basename(json_file))[0] + "_resized.json"
        new_json_path = os.path.join(output_dir, new_json_name)
        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return f"[完成] {img_name}"

    except Exception as e:
        return f"[错误] {json_file} → {e}"

# 获取全部 JSON 文件
json_files = glob(os.path.join(input_dir, "*.json"))
max_threads = min(32, os.cpu_count() * 2)

# 使用线程池并添加 tqdm 进度条
with ThreadPoolExecutor(max_threads) as executor:
    futures = {executor.submit(process_file, jf): jf for jf in json_files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度", ncols=100):
        print(future.result())

print("全部处理完成 ✅")
