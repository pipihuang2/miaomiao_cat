import os
import glob
import time
import random
import cv2
import tqdm
from PIL import Image
import shutil

class assy_tool:
    # 下面这个是检测是否又\t输出的
    def checkout_txt_label_right(self, output_file):
        with open(output_file, "r") as file:
            for line in file:
                parts = line.strip().split("\t")  # 按 \t 分割
                if len(parts) != 2:
                    print(f"行格式错误：{line}")
                else:
                    print(f"图片路径: {parts[0]}, 标签: {parts[1]}")

    # 下面这个是划分数据集txt的分别是train和val的
    def create_train_val_txt(self, val_pic_file, train_pic_file):
        with open(os.path.join(train_pic_file,'../rec_gt_train.txt'), 'w') as w:
            pic_list_1 = glob.glob(os.path.join(train_pic_file,'*.jpg'))
            if len(pic_list_1) == 0:
                print('train文件夹数据是空的')
                return
            for i in pic_list_1:
                absolute_file = os.path.join(i.rsplit('\\', 2)[1], i.rsplit('\\', 2)[2])
                label = os.path.basename(i).split('_')[0]
                w.write(f"{absolute_file}\t{label}\n")
                print(absolute_file)
        with open(os.path.join(val_pic_file,'../rec_gt_val.txt'), 'w') as w:
            pic_list_2 = glob.glob(os.path.join(val_pic_file,'*.jpg'))
            if len(pic_list_2) == 0:
                print('val文件夹数据是空的')
                return
            for i in pic_list_2:
                absolute_file = os.path.join(i.rsplit('\\', 2)[1], i.rsplit('\\', 2)[2])
                label = os.path.basename(i).split('_')[0]
                w.write(f"{absolute_file}\t{label}\n")
                print(absolute_file)



    def split_train_val_data(self, pic_list,train_doc='./train_',val_doc='./val_'):
        self.create_file(train_doc)
        self.create_file(val_doc)
        random.shuffle(pic_list)
        ration = 0.8
        split_index = int(len(pic_list) * ration)
        train = pic_list[:split_index]
        val = pic_list[split_index:]
        for i in train:
            try:
                shutil.move(i, train_doc)
            except shutil.Error as e:
                # 捕获目标文件已存在的错误
                print(f"目标文件已存{i}")
        for i in val:
            try:
                shutil.move(i, val_doc)
            except shutil.Error as e:
                # 捕获目标文件已存在的错误
                print(f"目标文件已存{i}")

    # 这个是改变图片名字的，后续用于分割数据用。
    def change_pic_name(self, pic_list):
        pic_number = len(pic_list)
        index = 0
        for i in tqdm.tqdm(pic_list):
            index += 1
            random_uniform = random.uniform(0.01, 0.9)
            image = Image.open(i)
            base_file = i.rsplit('\\', 1)[0] + f'/{i.rsplit('\\', 2)[1]}_{index}{pic_number * random_uniform}.jpg'
            image.save(base_file)
            image.close()
            os.remove(i)

    def create_file(self,file):
        os.makedirs(file, exist_ok=True)
        print(f"Directory '{file}' created successfully or already exists.")
    # #转换png为jpg,并删除原来的png数据
    def change_pic_label(self, pic_list, origin_label='.png', target_labe='.jpg'):
        if len(pic_list)==0:
            print(f'没有找到后缀为{origin_label}的文件')
            return
        if origin_label == target_labe:
            remove_pic = False
        else:
            remove_pic = True
        for i in tqdm.tqdm(pic_list):
            image = cv2.imread(i)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            i_del = i
            i = i.replace(origin_label, target_labe)
            cv2.imwrite(i, image)
            if remove_pic:
                os.remove(i_del)

    #todo 还没有一键完成
    def forward(self):
        pass

if __name__ == '__main__':
    hyj = assy_tool()
    pic_list_file = r'E:\25_01_13\move_move_\*.bmp'
    pic_list= glob.glob(pic_list_file)
    hyj.change_pic_label(pic_list, '.bmp')
    # hyj.change_pic_label(pic_list,'.png')
    # hyj.change_pic_name(pic_list)
    # hyj.split_train_val_data(pic_list,train_doc=r'F:\assy\train',val_doc=r'F:\assy\val')
    # hyj.create_train_val_txt(train_pic_file=r'F:\assy\train',val_pic_file=r'F:\assy\val')
    # hyj.checkout_txt_label_right(r'F:\assy\rec_gt_val.txt')