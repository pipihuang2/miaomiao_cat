import cv2, numpy as np

yaml_path = r"E:\hyy\miaomiao_cat\Calibration\stereo_params.yaml"

def load_stereo_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    w  = int(fs.getNode("image_width").real())
    h  = int(fs.getNode("image_height").real())
    K1 = fs.getNode("K1").mat(); D1 = fs.getNode("D1").mat()
    K2 = fs.getNode("K2").mat(); D2 = fs.getNode("D2").mat()
    R1 = fs.getNode("R1").mat(); R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat(); P2 = fs.getNode("P2").mat()
    Q  = fs.getNode("Q").mat()
    fs.release()
    return (w, h, K1, D1, K2, D2, R1, R2, P1, P2, Q)

w, h, K1, D1, K2, D2, R1, R2, P1, P2, Q = load_stereo_yaml(yaml_path)

# 生成可复用映射
map1A, map2A = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_16SC2)
map1B, map2B = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_16SC2)





# 示例：对一对新图 rectA/rectB
imgA = cv2.imread(r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\out\3\1861.jpg")
imgB = cv2.imread(r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\out\3\2258.jpg")
rectA = cv2.remap(imgA, map1A, map2A, cv2.INTER_LINEAR)
rectB = cv2.remap(imgB, map1B, map2B, cv2.INTER_LINEAR)
cv2.imwrite(r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\out\3\rectA.png", rectA)
cv2.imwrite(r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\out\3\rectB.png", rectB)

# 示例：SGBM 出视差 -> 用 Q 反投影
# sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
#                              P1=8*3*5*5, P2=32*3*5*5,
#                              disp12MaxDiff=1, uniquenessRatio=10,
#                              speckleWindowSize=100, speckleRange=2)
# disp = sgbm.compute(cv2.cvtColor(rectA, cv2.COLOR_BGR2GRAY),
#                     cv2.cvtColor(rectB, cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
# pts3d = cv2.reprojectImageTo3D(disp, Q)  # Z 单位与 T 一致
