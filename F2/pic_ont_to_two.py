import glob
import os
import cv2
import multiprocessing
from tqdm import tqdm
from functools import partial


def process_image(image_path, output_dir):
    """ 读取图片并分割成三部分 """
    pic_ = cv2.imread(image_path)
    if pic_ is None:
        print(f"Error reading {image_path}")
        return

    h, w = pic_.shape[:2]
    w_ = w // 2

    pic_1 = pic_[:, 0: w_]
    pic_2 = pic_[:, w_ : w_ * 2]


    base_name = os.path.basename(image_path).split('.jpg')[0]

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_1.jpg"), pic_1)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_2.jpg"), pic_2)



def main():
    input_folder = r"D:\Project\HYJ_Pic\F2\260112\three"
    output_folder = r"D:\Project\HYJ_Pic\F2\260112\two"
    os.makedirs(output_folder, exist_ok=True)

    pic_list = glob.glob(os.path.join(input_folder, "*.jpg"))

    # 使用 partial 传递额外参数
    process_func = partial(process_image, output_dir=output_folder)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_func, pic_list), total=len(pic_list)))


if __name__ == "__main__":
    main()
