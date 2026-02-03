import re
import yaml
import numpy as np
import cv2
collision_area_file = r"D:\Project\HYJ\git_vgs\vgs\config\mold\SF\YFK-C6-F-014\SF_collision_YFK-C6-F-014.yaml"
all_contours = []
with open(collision_area_file, "r", encoding="UTF-8") as y:
    yaml_content = yaml.safe_load(y)
    m = re.search(r"xst_start\s*(.*?)\s*xst_end", yaml_content, re.S)
    assert m, "没有找到 xst_start/xst_end 包围的内容"
    content = m.group(1)

    # 2) 以 ; 分割，并解析成 (x, y)
    pairs = []
    for s in content.split("\n"):
        pairs.append(s)


for s in pairs:
    pts = []
    # 去掉结尾多余的分号，然后按 ; 切
    for pair in s.strip().strip(";").split(";"):
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        pts.append([float(x_str), float(y_str)])  # 变成数字
    all_contours.append(pts)

data = {
    "contours": all_contours
}
print(pairs)
print(len(pairs))

with open("contours.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        data,
        f,
        allow_unicode=True,
        sort_keys=False
    )
# def to_px(p):
#     return (int(round(p[0] - min_x)) + margin,
#             int(round(p[1] - min_y)) + margin)
#
# with open(r"E:\hyy\miaomiao_cat\VGS\contours.yaml", "r", encoding="utf-8") as f:
#     out11 = yaml.safe_load(f)
#     out_pairs = {}
#
#     max_x = float("-inf")
#     min_x = float("inf")
#     max_y = float("-inf")
#     min_y = float("inf")
#
#     for index, s in enumerate(out11["contours"]):
#         pts = []
#         for s_ in s:
#             x, y = float(s_[0]), float(s_[1])
#             if x > max_x: max_x = x
#             if x < min_x: min_x = x
#             if y > max_y: max_y = y
#             if y < min_y: min_y = y
#             pts.append((x, y))
#         out_pairs[index] = pts
#         # 增加对应的边框
#     margin = 30
#     W = int(round(max_x - min_x)) + 2 * margin + 1
#     H = int(round(max_y - min_y)) + 2 * margin + 1
#     img = np.full((H, W, 3), 0, np.uint8)  # 黑底
#     mask1 = img.copy()
#     mask2 = img.copy()
#
#
#     # 4) 坐标平移到画布内（OpenCV 像素是整数）
#     def to_px(p):
#         return (int(round(p[0] - min_x)) + margin,
#                 int(round(p[1] - min_y)) + margin)
#
#
#     for index, s in enumerate(out_pairs):
#         pts = out_pairs[index]
#
#         int_pts = np.array([to_px(p) for p in pts], dtype=np.int32)
#
#         # 5) 画折线 & 每个点画小圆点
#         cv2.polylines(mask1, [int_pts], isClosed=False, color=(255, 0, 0), thickness=1)
#     import PIL.Image
#     PIL.Image.fromarray(mask1).show()
# print(out_pairs)