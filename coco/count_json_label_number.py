import json
from collections import Counter
import glob
import tqdm
# 获取所有 JSON 文件路径
json_list = glob.glob(r'F:\deeplearning\pytorch\miao_tools\miaomiao_cat\pic_and_json\total\*.json')

# 初始化全局计数器和类别集合
global_label_counts = Counter()
global_unique_labels = set()

# 遍历所有 JSON 文件
for json_file in tqdm.tqdm(json_list):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # 提取每个文件的类别
        categories = [item['label'] for item in data['shapes']]

        # 更新全局计数器和类别集合
        global_label_counts.update(categories)
        global_unique_labels.update(categories)

# 统计全局类别数量
global_category_count = len(global_unique_labels)

# 输出统计结果
print("类别数量统计：")
print(f"数据集中类别总数: {global_category_count}")
print(f"类别列表: {global_unique_labels}")
print("每个类别的实例数:")
for label, count in global_label_counts.items():
    print(f"{label}: {count}")
