import glob
import os.path

import cv2
import numpy as np
import tqdm
from qrdet import QRDetector
qr_detector = QRDetector(model_size="s")
postion={"region": [ 0.438619-0.5, 0.150351-0.5, 0.094688 ,0.110102 ]}
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

def detect_dm(image, params={},basename=None):
    # cv2.imshow('1',image)
    # cv2.waitKey(0)
    try:
        detections = qr_detector.detect(image=image, is_bgr=False)
        x1, y1, x2, y2 = detections[0]['bbox_xyxy']
        x1 = max(round(x1 - 5), 0)
        y1 = max(round(y1 - 5), 0)
        x2 = min(round(x2 + 5), image.shape[1])
        y2 = min(round(y2 + 5), image.shape[0])
        cropped_image = image[y1:y2, x1:x2]
        import random
        name = random.randint(1,9999999999)
        cv2.imwrite(fr'F:\qr\DAB\180B-S/{name}.jpg',cropped_image)
        # cv2.imshow('1',cropped_image)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    pic = glob.glob(r"U:\DAB\starfold\180B-S\250417\*.bmp")
    for i in tqdm.tqdm(pic):
        image = cv2.imread(i)
        image_crop = extract_region(image,postion)
        basename = os.path.basename(i)
        dm_pic = detect_dm(image_crop,basename)
