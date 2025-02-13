import os
import shutil
import random

# 源文件夹和目标文件夹路径
source_dir = r'F:\deeplearning\Train_image_data\star\241130NG'  # 存放原始文件的目录
train_dir = r'F:\deeplearning\Train_image_data\star\train'    # 训练集文件夹
val_dir = r'F:\deeplearning\Train_image_data\star\val'       # 验证集文件夹

# 创建目标文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有的 .bmp 文件
bmp_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

# 按 8:1 比例分配文件
for bmp_file in bmp_files:
    # 检查是否有对应的 .json 文件
    json_file = bmp_file.replace('.jpg', '.json')
    if json_file in os.listdir(source_dir):
        # 随机决定是否放入 train 或 val 文件夹
        target_dir = train_dir if random.random() < 0.8 else val_dir

        # 移动 .bmp 文件和对应的 .json 文件
        shutil.move(os.path.join(source_dir, bmp_file), os.path.join(target_dir, bmp_file))
        shutil.move(os.path.join(source_dir, json_file), os.path.join(target_dir, json_file))
        print(f"Moved {bmp_file} and {json_file} to {target_dir}")
    else:
        print(f"No matching .json file for {bmp_file}")
