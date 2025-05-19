import os
import numpy as np
import cv2

def scale_image(image, scale_percent=30):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image



def move(A, dx, dy, border_value=0):
    '''
    平移图像,并填充指定值
        A: 输入图像
        dx: x轴平移距离
        dy: y轴平移距离
        border_value: 填充值
    '''
    h, w = A.shape[:2]
    A_shifted = np.ones_like(A) * border_value
    if dx >= 0 and dy >= 0:
        A_shifted[dx:, dy:] = A[:h-dx, :w-dy]
    elif dx >= 0 and dy < 0:
        A_shifted[dx:, :w+dy] = A[:h-dx, -dy:]
    elif dx < 0 and dy >= 0:
        A_shifted[:h+dx, dy:] = A[-dx:, :w-dy]
    else:
        A_shifted[:h+dx, :w+dy] = A[-dx:, -dy:]
    return A_shifted

def move_to_centroid(image, border_value=255):
    x, y = np.where(image == 0)
    tx, ty = (int(x.mean()), int(y.mean()))
    h, w = image.shape[:2]
    cx, cy = h // 2, w // 2
    dx, dy = cx - tx, cy - ty
    return move(image, dx, dy, border_value=border_value), (dx, dy)


def get_standard_image(path,scale):
    scale = 4000/scale
    image_ = cv2.imread(path)
    _std = scale_image(image_, scale * 100)
    binary = cv2.threshold(cv2.cvtColor(_std, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1] ^ 255
    standard, _ = move_to_centroid(binary)
    cv2.imwrite('standard.png',standard)

if __name__ == '__main__':
    get_standard_image(r'D:\HYJJJ\cad\EUEA_3133.bmp',3133)