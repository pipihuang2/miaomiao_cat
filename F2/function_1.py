import cv2

def resize_flip_image(image_file: str, pix_1000_: float, flip: bool=False) -> None:
    pix_1000_scale = pix_1000_ / 1000 #这里是标准1000mm的线对应多少像素
    resize_scale = 4 / pix_1000_scale  #我们通过标定板标定后的图片都是4个像素1mm

    image_ = cv2.imread(image_file)
    if flip:
        image_ = cv2.flip(image_, 1)
    image_ = cv2.resize(image_, (0, 0),fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(image_file, image_)

if __name__ == '__main__':
    resize_flip_image(r"D:\Project\HYJ_Pic\F2\cad_out\1\20260721-141228.png",2355)