
import cv2
from qrdet import QRDetector
import glob
import os
import shutil
qr_detector = QRDetector(model_size="s")
out_pic = r"F:\DAB\front"
pic = glob.glob("F:\DAB\*.jpg")
for image in pic:
    file_name = image
    image = cv2.imread(image)
    image = image[221:379,1012:1281]
    detections = qr_detector.detect(image=image, is_bgr=False)
    if len(detections) !=0:
        x1, y1, x2, y2 = detections[0]['bbox_xyxy']
        x1 = max(round(x1 - 5), 0)
        y1 = max(round(y1 - 5), 0)
        x2 = min(round(x2 + 5), image.shape[1])
        y2 = min(round(y2 + 5), image.shape[0])
        cropped_image = image[y1:y2, x1:x2]
        cv2.imwrite(f'F:/qr/{os.path.basename(file_name)}',cropped_image)
        shutil.move(file_name,out_pic)