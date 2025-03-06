import json
import os

# 设置源 JSON 文件夹和输出 COCO JSON 文件
source_dir = r'F:\800B\val'  # JSON 文件所在的目录
output_file = r'F:\800B\coco_format_val.json'

# 初始化 COCO 格式的字典
coco_format = {
    "info": {
        "description": "Example Dataset",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "",
        "date_created": "2024-10-28"
    },
    "licenses": [
        {
            "id": 1,
            "name": "CC-BY",
            "url": "https://creativecommons.org/licenses/by/4.0/"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": []
}

# 计数器
image_id = 0
annotation_id = 0
category_dict = {}  # 用于存储类别和其对应的ID

# 定义标签映射，将中文标签转换为英文
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

# 遍历 source_dir 中的所有 JSON 文件
for filename in os.listdir(source_dir):
    if filename.endswith('.json'):
        with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取图像信息并添加到 COCO 字典中
        image_id += 1
        image_info = {
            "id": image_id,
            "file_name": data["imagePath"],
            "width": data["imageWidth"],
            "height": data["imageHeight"]
        }
        coco_format["images"].append(image_info)

        # 提取注释信息并添加到 COCO 字典中
        for shape in data["shapes"]:
            # 获取类别名称并转换为英文
            category_name = shape.get("label", "unknown")
            category_name = label_mapping.get(category_name, category_name)  # 转换中文为英文，如果没有映射则保持原样

            # 如果类别不存在于字典中，则添加到categories，并分配一个新的类别ID
            if category_name not in category_dict:
                category_id = len(category_dict)
                category_dict[category_name] = category_id
                category_info = {
                    "id": category_id,
                    "name": category_name,
                    "supercategory": "defect"
                }
                coco_format["categories"].append(category_info)

            # 获取类别ID
            category_id = category_dict[category_name]

            if shape["shape_type"] == "rectangle":
                # 计算 bounding box: [x_min, y_min, width, height]
                x_min = min(point[0] for point in shape["points"])
                y_min = min(point[1] for point in shape["points"])
                width = max(point[0] for point in shape["points"]) - x_min
                height = max(point[1] for point in shape["points"]) - y_min

                # 创建注释信息
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # 使用对应的类别ID
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "segmentation": [],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

# 保存为 COCO 格式的 JSON 文件
with open(output_file, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"转换完成，所有文件已合并到 '{output_file}'")
