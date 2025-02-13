import json
import os
import shutil

# 目标数量要求
required_cut_count = 17
required_assy_count = 1

# 定义文件夹路径
input_folder = r"G:\250217_star\YDB"  # 替换为你的输入文件夹路径
output_folder = r"G:\250217_star\no_good"+f"/{os.path.basename(input_folder)}"  # 替换为你想存放文件的目标文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)


def process_yaml(yaml_file_path):
    # 读取 YAML 文件
    with open(yaml_file_path, 'r') as f:
        data = json.load(f)

    # 统计 cut 和 assy 的数量
    cut_count = sum(1 for shape in data["shapes"] if shape["label"] == "cut")
    assy_count = sum(1 for shape in data["shapes"] if shape["label"] == "assy")

    # 打印统计信息
    print(f"cut count: {cut_count}, assy count: {assy_count}")

    # 如果数量不符合要求，则移动图片和 YAML 文件
    if cut_count != required_cut_count or assy_count != required_assy_count:
        image_path = data["imagePath"]
        # 获取文件的完整路径
        image_file_path = os.path.join(input_folder, image_path)

        # 移动图片和 YAML 文件
        shutil.move(image_file_path, os.path.join(output_folder, image_path))
        shutil.move(yaml_file_path, os.path.join(output_folder, os.path.basename(yaml_file_path)))
        print(f"Moved {image_path} and {yaml_file_path} to {output_folder}")


# 假设你已经有一个文件夹包含所有的 YAML 文件
for yaml_file in os.listdir(input_folder):
    if yaml_file.endswith(".json"):  # 如果是 JSON 格式的 YAML 文件
        yaml_file_path = os.path.join(input_folder, yaml_file)
        process_yaml(yaml_file_path)
