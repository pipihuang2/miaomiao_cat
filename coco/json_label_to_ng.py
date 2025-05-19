import os
import json
from glob import glob

# 中文 → 英文标签映射
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

# 输入和输出路径
input_dir = r"E:\yrt\kong\data_0425\val"         # 原始 JSON 文件夹
output_dir = r"E:\yrt\kong\data_0425\val"    # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 扫描所有 JSON 文件
json_files = glob(os.path.join(input_dir, "*.json"))

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 替换每个标注里的 label
    for shape in data.get("shapes", []):
        original_label = shape.get("label", "")
        if original_label in label_mapping:
            shape["label"] = label_mapping[original_label]

    # 保存修改后的 JSON
    new_json_path = os.path.join(output_dir, os.path.basename(json_path))
    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] 转换完成: {json_path}")

print("全部完成 ✅")
