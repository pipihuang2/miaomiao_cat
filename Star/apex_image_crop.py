import os.path
import tqdm
import cv2
import tqdm
from ultralytics import YOLO
import numpy as np
import glob

#这个是用来crop图片的
def extract_region(image, position, mask=False):
    if 'region' in position:
        x, y, tx, ty = position['region']
    else:
        x, y, tx, ty = position
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    x = int(x * w) + cx
    y = int(y * h) + cy
    tx = int(tx * w) // 2
    ty = int(ty * h) // 2
    if mask:
        background = np.zeros(image.shape, np.uint16)
        background[y - ty: y + ty, x - tx: x + tx] = image[y - ty: y + ty, x - tx: x + tx]
        return background
    return image[y - ty: y + ty, x - tx: x + tx]


if __name__ == '__main__':




    def compute_centroid(points):
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (int(x_sum / len(points)), int(y_sum / len(points)))

    model = YOLO(r"E:\hyy\miaomiao_cat\Star\models\best.onnx")
    pic = glob.glob(r"E:\STAR_NG\pic_all\*.jpg")
    for i in pic:
        name = os.path.basename(i)
        imagee = cv2.imread(i)
        out_image=extract_region(imagee,[ 0, -0.4, 0.08, 0.1])
        out_image=cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
        alpha = 1.2
        beta = 0
        adjusted_image = cv2.convertScaleAbs(out_image, alpha=alpha, beta=beta)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(adjusted_image)
        cv2.imwrite(os.path.join("E:\STAR_NG\line",name),img)

    #breakpoint()
    #下面是用来校正的
    # for i in tqdm.tqdm(pic):
    #     out_index=model(i)
    #     image = cv2.imread(i)
    #     cx, cy = image.shape[1] // 2, image.shape[0] // 2
    #     h, w = image.shape[:2]
    #     if len(out_index[0].boxes.conf) == 4:
    #         xywh_numpy = out_index[0].boxes.xywh.to("cpu").numpy()
    #         points = []
    #         for xy in xywh_numpy:
    #             points.append(xy[:2])
    #         centroid=compute_centroid(points)
    #         dx, dy = cx-centroid[0], cy-centroid[1]
    #         M = np.float32([[1, 0, dx],
    #                         [0, 1, dy]])
    #         dst = cv2.warpAffine(image, M, (w, h))
    #         # dst = dst[1248:1499, 398:616]
    #         #dst = dst[1569:1747,1437:1616] star
    #         cv2.imwrite(i,dst)
    #         #cv2.imwrite(os.path.join(r"E:\apex\crop_apex\ok",os.path.basename(i).replace('.jpg','.jpg')),dst)

