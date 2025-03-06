import json
import glob
import shutil
import tqdm

json_file = glob.glob(r'G:\123345\COMBINED222\three\*.json')
valid_json_files = []
for i in tqdm.tqdm(json_file):
    with open(i, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 提取并统计标签信息
        if len(data.get("shapes")) == 0:
            pass
        else:
            valid_json_files.append(i)
            for shape in data.get("shapes", []):
                label = shape.get("label", "unknown")


for i in tqdm.tqdm(valid_json_files):
    shutil.move(i,r'G:\123345\COMBINED222\homeaaa')
    i=i.replace('.json','.jpg')
    shutil.move(i, r'G:\123345\COMBINED222\homeaaa')