import numpy as np
import cv2


def transform_coords(coords, old_bg_size, new_bg_size):
    """
    将旧背景上的坐标点映射到新背景
    :param coords: [(x,y),...] 或 numpy 数组 (N,2)，坐标基于旧背景
    :param old_bg_size: (W1, H1) 旧背景尺寸
    :param new_bg_size: (W2, H2) 新背景尺寸
    :return: numpy 数组 (N,2)，映射后的新背景坐标
    """
    coords = np.asarray(coords, dtype=float)

    W1, H1 = old_bg_size
    W2, H2 = new_bg_size

    # 旧背景和新背景的中心
    cx1, cy1 = W1 / 2.0, H1 / 2.0
    cx2, cy2 = W2 / 2.0, H2 / 2.0

    # 把点坐标转换到“相对中心”的坐标系
    relative = coords - np.array([cx1, cy1])

    # 如果没有缩放：直接平移
    new_coords = relative + np.array([cx2, cy2])

    return new_coords


if __name__ == '__main__':
    import json

    # E:\hyy\miaomiao_cat\F2\CAB_F2_20250818_201243186_1_combined_loader.jpg

    origin_pic = cv2.imread(r'E:\hyy\miaomiao_cat\F2\CAB_F2_20250816_000259395_1_combined_loader.jpg')
    now_pic = cv2.imread(r'D:\Project\HYJ\cosmos\check\F2\standard\uge.jpg')
    # new_points = transform_coords([(4000,600),(4800,1300)], (origin_pic.shape[1],origin_pic.shape[0]), (now_pic.shape[1],now_pic.shape[0])).reshape(-1)
    # print(new_points.reshape(-1))
    # new_points = np.asarray(new_points,dtype=np.uint32)
    # x1, y1, x2, y2 = new_points

    roi = now_pic[199:700, 5179:5821]
    # roi = origin_pic[50:550, 4160:4700]
    cv2.imshow('1',roi)
    cv2.waitKey(0)

    # with open(r"D:\Project\HYJ\cosmos\check\F2\D01\0818\UGE.json","r",encoding="UTF-8") as r:
    #     json_content = json.load(r)
    #     Points = json_content['shapes'][0].get("points")
    #     new_points = transform_coords(Points, (origin_pic.shape[1], origin_pic.shape[0]),
    #                                   (now_pic.shape[1], now_pic.shape[0]))
    #     print(new_points)
    #     roi_list = []
    #     for point in new_points:
    #         roi_list.append([int(point[0]),int(point[1])])
    #     pts = np.array(roi_list,dtype=np.int32).reshape((-1, 1, 2))
    #     cv2.polylines(now_pic, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    #     import PIL.Image
    #     PIL.Image.fromarray(now_pic).show()
    #
    #
