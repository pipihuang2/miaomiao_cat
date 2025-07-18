import glob
import os
import shutil
import pprint

# pic_file_lis = glob.glob(r'pic\*.jpg')
# target_file = r'F:\deeplearning\pytorch\miao_tools\miaomiao_cat'
# for file in pic_file_lis:
#     shutil.move(file,target_file)
#     print('完成传送')


# 下面这个函数是把标定后的文件不如说pic或者xml各自都放到一个单独的文件夹里
# 我下面就是先找出xml文件，然后把xml文件对应的图片找出来
json_file_list = glob.glob(r'F:\deeplearning\pytorch\miao_tools\miaomiao_cat\pic\*.json')
pic_file_list = glob.glob(r'F:\deeplearning\pytorch\miao_tools\miaomiao_cat\pic\*.jpg')
if not json_file_list and not pic_file_list:
    print('json路径中没有相关文件,pic路径中没有相关图片')
elif not pic_file_list:
    print('pic路径中没有图片')
elif not json_file_list:
    print('json路径中没有相关文件')

target_file_json = 'D:\Project\HYJ\Data_Processing\json'
target_file_jpg = 'D:\Project\HYJ\Data_Processing2\jpg'

different_file_list = [os.path.basename(file).split('.')[0] for file in json_file_list]
same_file_list = [os.path.basename(file).split('.')[0] for file in json_file_list]
original_json_list = [os.path.join(r'D:\Project\HYJ\Data_Processing\241120', file + '.json') for file in
                      different_file_list]
original_pic_list = [os.path.join(r'D:\Project\HYJ\Data_Processing\241120', file + '.jpg') for file in
                     different_file_list]
pprint.pprint(original_json_list)
for i in original_json_list:
    shutil.move(i, target_file_json)

for i in original_pic_list:
    shutil.move(i, target_file_jpg)
