import os
import glob
import random
import shutil
from typing import List, Tuple, Sequence

import tqdm


def split_ratio(lst: Sequence,ration: Tuple[int,int]) -> tuple[list,list]:
    rng = random.Random()
    idx = list(range(len(lst)))
    rng.shuffle(idx)
    n1 = round(len(lst) * ration[0] / (ration[0] + ration[1]))
    part1 = [lst[i] for i in idx[:n1]]
    part2 = [lst[i] for i in idx[n1:]]
    return part1, part2


dir_file = r"C:\Users\Administrator\Desktop\label_ng\20260110\train\e"
train_dir = os.path.join(dir_file, "train")
valid_dir = os.path.join(dir_file, "val")
out = os.listdir(dir_file)
for item in tqdm.tqdm(out):
    original_file = os.path.join(dir_file, item)
    pic_list = glob.glob(os.path.join(original_file, "*.jpg")) + glob.glob(os.path.join(original_file, "*.png"))
    part1, part2 = split_ratio(pic_list, (8,2))
    train_dir_file = os.path.join(train_dir, item)
    valid_dir_file = os.path.join(valid_dir, item)
    for i1 in part1:
        os.makedirs(train_dir_file,exist_ok=True)
        shutil.move(i1, train_dir_file)
    for i2 in part2:
        os.makedirs(valid_dir_file,exist_ok=True)
        shutil.move(i2, valid_dir_file)