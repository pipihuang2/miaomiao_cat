import os.path

import cv2
import numpy as np
import glob

import tqdm


def _execute(image, mask, fill_value) -> np.ndarray:
    """根据掩码提取 ROI 区域"""
    # 创建背景并填充指定颜色
    output_img = np.zeros_like(image)
    output_img[...] = fill_value

    # 应用掩码
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    roi = mask > 0
    output_img[roi] = image[roi]

    return output_img

mask_pic = cv2.imread(r"D:\Project\HYJ_Pic\vgs\total_pic\ng\1112.png")
pic_list = glob.glob(r"D:\Project\HYJ_Pic\vgs\total_pic\251217\**\*original_image.png", recursive=True)
# pic_list2 = glob.glob(r"D:\Project\HYJ_Pic\vgs\total_pic\202511\**\original_image_offset.png", recursive=True)
total_pic = pic_list
for index, pic_path in tqdm.tqdm(enumerate(total_pic), total=len(total_pic)):
    image = cv2.imread(pic_path)
    output_img = _execute(image, mask_pic, 255)
    import datetime
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cv2.imwrite(fr"D:\Project\HYJ_Pic\vgs\total_pic\ng\1\{time}_{index}.png", output_img)