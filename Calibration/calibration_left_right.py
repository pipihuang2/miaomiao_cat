import cv2, glob, numpy as np

# 1) 读取你的内参
fs = cv2.FileStorage(r"E:\hyy\miaomiao_cat\Calibration\out\intrinsics.yaml", cv2.FILE_STORAGE_READ)
K    = fs.getNode("camera_matrix").mat().astype(np.float64)
dist = fs.getNode("distortion_coefficients").mat().astype(np.float64)
fs.release()   # 你这份yaml没有畸变，就先设为0；如有请替换

# 2) 若双目标定/数据采集分辨率和 5472x3648 不一致，先按比例缩放K
def scale_K(K, old_size, new_size):
    ow, oh = old_size
    nw, nh = new_size
    sx, sy = nw/ow, nh/oh
    K2 = K.copy().astype(np.float64)
    K2[0,0] *= sx;  K2[1,1] *= sy
    K2[0,2] *= sx;  K2[1,2] *= sy
    return K2

# 3) 读棋盘对
pattern_size = (9,6)     # 按你的棋盘内角点
square_size  = 290    # 每格 25mm
lefts  = sorted(glob.glob(r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\0919\1141/*.bmp"))
rights = sorted(glob.glob(r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\0919\1541/*.bmp"))
assert len(lefts)==len(rights) and len(lefts)>=8

objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2) * square_size

objpoints, imgpointsL, imgpointsR = [], [], []
crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

for pL, pR in zip(lefts, rights):
    gL = cv2.imread(pL, cv2.IMREAD_GRAYSCALE)
    gR = cv2.imread(pR, cv2.IMREAD_GRAYSCALE)

    retL, cL =  cv2.findChessboardCornersSB(
        gL, pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER
    )
    retR, cR = cv2.findChessboardCornersSB(gR, pattern_size,flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER)
    if retL and retR:
        cL = cv2.cornerSubPix(gL, cL, (11,11), (-1,-1), crit)
        cR = cv2.cornerSubPix(gR, cR, (11,11), (-1,-1), crit)
        objpoints.append(objp.copy())
        imgpointsL.append(cL); imgpointsR.append(cR)

print(f"[Info] 有效棋盘对: {len(objpoints)}")

h, w = cv2.imread(lefts[0], cv2.IMREAD_GRAYSCALE).shape[:2]


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
flags = ( 0
        # 典型可选：
        # | cv2.CALIB_ZERO_TANGENT_DIST
        # | cv2.CALIB_RATIONAL_MODEL
        # | cv2.CALIB_SAME_FOCAL_LENGTH
        )
# flags = cv2.CALIB_FIX_INTRINSIC  # 关键：固定内参/畸变 就是用自己算出来的畸变和内参

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    None, None, None, None,
    (w,h),
    criteria=criteria, flags=flags
)

print("RMS reprojection error:", ret)
print("K1=\n", K1, "\nD1=", D1.ravel())
print("K2=\n", K2, "\nD2=", D2.ravel())
print("R=\n", R)
print("T=\n", T, "  |baseline|=", np.linalg.norm(T))


#上面可以求出baseline基线

#下面就是要做立体校正，校正两个相机的极限

# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#     K1, D1, K2, D2, image_size, R, T,
#     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
# )

# map1x, map1y = cv2.initUndistortRectifyMap(
#     K1, D1, R1, P1, image_size, cv2.CV_32FC1)
# map2x, map2y = cv2.initUndistortRectifyMap(
#     K2, D2, R2, P2, image_size, cv2.CV_32FC1)


# rectL = cv2.remap(imgL, map1x, map1y, interpolation=cv2.INTER_LINEAR)
# rectR = cv2.remap(imgR, map2x, map2y, interpolation=cv2.INTER_LINEAR)








#这个是算假设 A视角中的物体C 在B视角中满足的极线方程 算一下视角B中特征点 到A中理想情况下极限的距离
def point_to_epiline_distance(F, ptA, ptB):
    """
    F: 3x3 基础矩阵（从A到B）
    ptA: (u, v) A图像上的像素点
    ptB: (u, v) B图像上的像素点
    返回:
      d_B: 以A点生成的B中极线，ptB到该极线的垂直距离（像素）
      d_A: 以B点生成的A中极线，ptA到该极线的垂直距离（像素）
    """
    xA = np.array([ptA[0], ptA[1], 1.0], dtype=np.float64)
    xB = np.array([ptB[0], ptB[1], 1.0], dtype=np.float64)

    # B中的极线: l' = F xA
    lB = F @ xA      # [a, b, c]
    a, b, c = lB
    d_B = abs(a*ptB[0] + b*ptB[1] + c) / np.sqrt(a*a + b*b + 1e-12)

    # A中的极线: l = F^T xB
    lA = F.T @ xB
    a, b, c = lA
    d_A = abs(a*ptA[0] + b*ptA[1] + c) / np.sqrt(a*a + b*b + 1e-12)

    return d_B, d_A



#
#
#
# # 如果这批图片不是 5472x3648，请用 scale_K 调整
# K1 = scale_K(K, (5472,3648), (w,h)) if (w,h)!=(3648,5472) and (w,h)!=(3648,5472) else K.copy()
# K2 = K1.copy()
# D1, D2 = D.copy(), D.copy()
#
# # stereoCalibrate：固定内参，只估计外参 R,T,E,F
# criteria_st = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-5)
# flags = cv2.CALIB_FIX_INTRINSIC
# rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, (w,h),
#     criteria=criteria_st, flags=flags
# )
#
# print("RMS:", rms)
# print("R:\n", R)
# print("T:", T.ravel(), " | 基线B=", np.linalg.norm(T))
#
# # 立体校正映射
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1,D1,K2,D2,(w,h),R,T,flags=cv2.CALIB_ZERO_DISPARITY)
# print("Q:\n", Q)
#
#
# # ------- 小笔误修正：分辨率判断 -------
# # 如果这批图片不是 5472x3648，则等比缩放 K；否则直接用原 K
# if (w, h) != (3648, 5472):
#     K1 = scale_K(K, (5472, 3648), (w, h))
# else:
#     K1 = K.copy()
# K2 = K1.copy()
# D1, D2 = D.copy(), D.copy()
#
# # ------- stereoCalibrate：固定内参，只估计外参 -------
# criteria_st = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
# flags = cv2.CALIB_FIX_INTRINSIC
# rms, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, (w, h),
#     criteria=criteria_st, flags=flags
# )
# print("RMS:", rms)
# print("R:\n", R)
# print("T:", T.ravel(), " | 基线B=", np.linalg.norm(T))
#
# # ------- 立体校正映射 -------
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#     K1o, D1o, K2o, D2o, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY
# )
# print("Q:\n", Q)
#
# # ------- 保存到 YAML -------
# def save_stereo_yaml(path):
#     fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
#     fs.write("image_width", int(w))
#     fs.write("image_height", int(h))
#     fs.write("K1", np.asarray(K1o, dtype=np.float64))
#     fs.write("D1", np.asarray(D1o, dtype=np.float64))
#     fs.write("K2", np.asarray(K2o, dtype=np.float64))
#     fs.write("D2", np.asarray(D2o, dtype=np.float64))
#     fs.write("R",  np.asarray(R,  dtype=np.float64))
#     fs.write("T",  np.asarray(T,  dtype=np.float64))
#     fs.write("R1", np.asarray(R1, dtype=np.float64))
#     fs.write("R2", np.asarray(R2, dtype=np.float64))
#     fs.write("P1", np.asarray(P1, dtype=np.float64))
#     fs.write("P2", np.asarray(P2, dtype=np.float64))
#     fs.write("Q",  np.asarray(Q,  dtype=np.float64))
#     # ROI: [x, y, w, h]
#     fs.write("roi1", np.array([roi1[0], roi1[1], roi1[2], roi1[3]], dtype=np.int32))
#     fs.write("roi2", np.array([roi2[0], roi2[1], roi2[2], roi2[3]], dtype=np.int32))
#     fs.release()
#     print(f"[Info] stereo params saved -> {path}")
#
# save_stereo_yaml(r"E:\hyy\miaomiao_cat\Calibration\stereo_params.yaml")