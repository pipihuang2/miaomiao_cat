import glob
import tqdm
import cv2
from ultralytics import YOLO
import shutil
import pathlib

if __name__ == '__main__':
    model = YOLO(r'E:\hyy\miaomiao_cat\Star\models\front_back.pt')  # 选择nano版本

    # model = YOLO(r'E:\1.6\classify\runs\classify\train7\weights\last.pt', task='classify', verbose=False)
    pic = glob.glob(r'E:\ng_f2\front\*.jpg')
    pic = list(pathlib.Path("U:\DAB\starfold").rglob("*.bmp"))
    print(f'一共有{len(pic)}张图片')
    for i in tqdm.tqdm(pic):
        image = cv2.imread(i)
        out_=model(image,  imgsz=224, verbose=False)
        # print(out_[0].probs.top1)
        if out_[0].probs.top1 == 0 :
            shutil.copy(i,r"F:\DAB")

    # model.export(format="onnx")
    # # 开始训练
    #model.train(data=r'E:\1.6\classify\CAB_front', epochs=100, imgsz=224)