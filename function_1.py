import glob
import shutil

pic_file_lis = glob.glob(r'pic\*.jpg')
target_file = r'F:\deeplearning\pytorch\miao_tools\miaomiao_cat'
for file in pic_file_lis:
    shutil.move(file,target_file)