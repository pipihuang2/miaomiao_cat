import cv2
import glob
import tqdm
from multiprocessing import Pool
def shrink_image(i):
    img = cv2.imread(i)
    if img is None:
        print(f"Failed to read {i}")
        return
    cv2.imwrite(i, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])


if __name__ == '__main__':
    pic = glob.glob(r"F:\CB6FRONT\CB6FRONTcombine\*.jpg")
    with Pool() as pool:
        list(tqdm.tqdm(pool.imap(shrink_image, pic), total=len(pic), desc="pic_number"))
