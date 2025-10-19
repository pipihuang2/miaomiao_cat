import json
import math
import os.path


#用来逆时针排序点的
def sort_points_counter_clockwise(points):
    # 如果点少于2个，直接返回原列表
    if len(points) < 2:
        return points

    # 计算质心
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)

    # 按角度排序（逆时针）
    def get_angle(point):
        return math.atan2(point[1] - centroid_y, point[0] - centroid_x)

    # 返回排序后的新列表，角度从大到小（逆时针）
    return sorted(points, key=get_angle, reverse=True)
import glob
json_files = glob.glob(r"E:\hyy\miaomiao_cat\F2\json\1\*.json")
out_data = {}

for json_file in json_files:
    with open(json_file,'r',encoding='utf-8') as file:
        data = json.load(file)
        base_name = os.path.basename(json_file).split('.')[0]
        out_data.setdefault(base_name,'None')
        shape = data.get("shapes",None)
        out_list = []
        if len(shape) % 5 ==0:
            for i in range(0,len(shape),5):
                combine_ = []
                four_location_point = []
                roi_points = shape[i].get("points")
                # roi_points = sort_points_counter_clockwise(roi_points)
                result = [list(map(int,roi_point)) for roi_point in roi_points]
                for i_ in range(1,5,1):
                    point_=shape[i+i_].get("points")[0]
                    four_location_point.append(list(map(int,point_)))
                print('我是四个点',four_location_point)
                combine_.append(result)
                combine_.append(four_location_point)
                # print(combine_)
                out_list.append(combine_)
            print(out_list)
            out_data[base_name] = out_list

        else:
            print("ROI区域和四个点个数对不上（总label必须是5的倍数）")


with open("out_roi.json","w",encoding="utf-8") as file_write:
    json.dump(out_data,file_write,ensure_ascii=False)
#
# print(shape)