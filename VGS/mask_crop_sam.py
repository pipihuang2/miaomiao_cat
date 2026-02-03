import glob
import os.path
import numpy as np
import cv2

image = cv2.imread(r"D:\Project\HYJ\git_vgs\vgs\output\result_251025_150710\hyj6666666666.jpg")

def mask_to_256(image_file: str, out_file: str):
    # 读取原始大图
    image = cv2.imread(image_file)
    orig = image
    h, w, _ = orig.shape
    tile_size = 256
    center_size = 196
    # 计算中心区域大小
    margin = (tile_size - center_size) // 2

    # 计算需要填充的尺寸
    pad_h = ((int(h / center_size) + 1) * center_size) - h
    pad_w = ((int(w / center_size) + 1) * center_size) - w
    img_pad = cv2.copyMakeBorder(orig, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_pad = cv2.copyMakeBorder(img_pad, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    H, W, _ = img_pad.shape
    print(f"原始图像尺寸: {h}x{w}，填充后图像尺寸: {H}x{W}")
    # 预计算所有tile的坐标
    y_coords = range(0, H - center_size, center_size)
    x_coords = range(0, W - center_size, center_size)
    hyj = 0
    base_name = os.path.basename(image_file)
    for y in y_coords:  # 未来需要优化，检测哪些区域需要处理
        for x in x_coords:
            tile = img_pad[y:y + tile_size, x:x + tile_size]
            hyj += 1
            cv2.imwrite(os.path.join(out_file,f"{hyj}"+base_name),tile)
if __name__ == '__main__':
    pic = glob.glob(r"D:\Project\HYJ_Pic\vgs\total_pic\ng\*.png")
    for f in pic:
        mask_to_256(f,r"D:\Project\HYJ_Pic\vgs\out_256")
