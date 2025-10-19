import json
import os.path

import cv2
import tqdm


def crop_image(json_path,index2):
    # 读取 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 读取图像
    image_path = data['imagePath']  # 确保图像在相同路径下
    image = cv2.imread(os.path.join(r"D:\Project\HYJ_Pic\20240718_cab\coco\val",image_path))

    if image is None:
        raise FileNotFoundError(f"图像未找到: {image_path}")

    # 遍历 shapes（可以多个 shape）
    for i, shape in enumerate(data['shapes']):
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            # 获取左上角和右下角坐标
            x1, y1 = map(int, points[0])
            x2, y2 = map(int, points[2])
            # 裁剪图像
            cropped = image[y1:y2, x1:x2]
            # 保存抠图
            output_path = fr'D:\Project\HYJ_Pic\20240718_cab\coco\out\cropped_val_{index2}_{i}_.jpg'
            try:
                cv2.imwrite(output_path, cropped)
            except Exception as T:
                print(T)
            # print(f"保存裁剪图像：{output_path}")

if __name__ == '__main__':
    import glob
    json_list = glob.glob(r"D:\Project\HYJ_Pic\20240718_cab\coco\val\*.json")
    for index,json_ in tqdm.tqdm(enumerate(json_list)):
        crop_image(json_,index)

