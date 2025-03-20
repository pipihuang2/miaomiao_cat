import cv2
import os
import shutil
import glob
pic = glob.glob(r"G:\0319\COMBINED\*.jpg")
for i in range(0,len(pic),2):
    pic_list = [pic[i+i_] for i_ in range(1)]
    for i__ in pic_list:
        shutil.move(i__,r"G:\0319\2")



