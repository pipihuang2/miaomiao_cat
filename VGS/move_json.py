import json
import glob
import shutil

import tqdm

out_file = r"D:\Project\HYJ_Pic\F2\true_have\1\1111"
pic_json = glob.glob(r'D:\Project\HYJ_Pic\F2\true_have\old\coco\coco\train/*.json')
i2 = 0
pic_filee = []
json_filee = []
for pic in tqdm.tqdm(pic_json):
    i2 += 1
    with open(pic,'r',encoding='utf-8') as f:
        pic_data = json.load(f)
        if len(pic_data.get('shapes', [])) > 0:
            shapes = pic_data.get('shapes', [])[0].get('points')
            pic_file = pic.replace('.json','.jpg')

            if len(shapes) > 1:
                try:
                    pic_filee.append(pic_file)
                    json_filee.append(pic)
                    # shutil.move(pic_file, out_file)
                    # shutil.move(pic, out_file)
                except Exception as e:
                    print(e)

for json_single in tqdm.tqdm(json_filee):
    shutil.move(json_single, out_file)
for pic_single in tqdm.tqdm(pic_filee):
    shutil.move(pic_single, out_file)