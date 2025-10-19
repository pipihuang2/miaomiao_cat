import cv2
import numpy as np
import glob
import os
import tqdm
import csv

# ======= 配置（按需修改）=======
images_glob  = r"E:\hyy\miaomiao_cat\Calibration\calibration_pic\0922/*.bmp"  # 标定图片通配符
pattern_size = (9, 6)  # 内角点数量 (列, 行)
square_size  = 1.0     # 单格边长；内参不依赖绝对单位，这里用1.0更省心
show_detect  = False    # 是否可视化角点检测
out_dir      = r"E:\hyy\miaomiao_cat\Calibration\out"
os.makedirs(out_dir, exist_ok=True)
# =================================

# 生成棋盘在世界坐标的3D点（Z=0平面）
cols, rows = pattern_size
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

# 角点检测精化参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

objpoints = []  # 3D点
imgpoints = []  # 2D点
good_paths = []

paths = sorted(glob.glob(images_glob))
if not paths:
    raise FileNotFoundError(f"未找到标定图片：{images_glob}")

print(f"[Info] 共发现 {len(paths)} 张图片，开始提取角点 ...")
for p in tqdm.tqdm(paths):
    img = cv2.imread(p)
    name = os.path.split(p)[-1]
    if img is None:
        print(f"[Warn] 读取失败: {p}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 先尝试 FAST_CHECK 提速；若失败，去掉 FAST_CHECK 再试一次

    # ret, corners = cv2.findChessboardCorners(
    #     gray, pattern_size,
    #     flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    # )

    ret, corners = cv2.findChessboardCornersSB(
        gray, pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER
    )

    if not ret:
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        good_paths.append(p)

        if show_detect:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            vis_show = cv2.resize(vis, None, fx=0.1, fy=0.1)
            cv2.imshow("Corners", vis_show)
            cv2.imwrite(os.path.join(out_dir, name), vis)
            cv2.waitKey(150)
    else:
        print(f"[Warn] 未检测到内角点: {p}")

if show_detect:
    cv2.destroyAllWindows()

if len(objpoints) < 5:
    raise RuntimeError(f"可用图片太少：{len(objpoints)}（建议 ≥ 5）")

# 使用第一张有效图片的尺寸
sample = cv2.imread(good_paths[0])
h, w = sample.shape[:2]

print(f"[Info] 使用 {len(objpoints)} 张有效图片进行标定 ...")
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)


# 保存内参到 YAML
out_yaml = os.path.join(out_dir, "intrinsics.yaml")
fs = cv2.FileStorage(out_yaml, cv2.FILE_STORAGE_WRITE)
fs.write("image_width", w)
fs.write("image_height", h)
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.release()
print(f"[Info] 内参已保存: {out_yaml}")






# ===== 计算 per-view 重投影误差 =====
per_view_stats = []  # [(path, n_points, rms, median, max), ...]
total_err2 = 0.0
total_pts = 0

for i in range(len(objpoints)):
    # 重新投影
    reproj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)  # (N,1,2)
    # 逐点像素误差
    diff = imgpoints[i] - reproj                     # (N,1,2)
    err_pt = np.sqrt((diff ** 2).sum(axis=2))        # (N,1) 每个点的欧氏距离
    # 每张图的 RMS：sqrt(mean(err^2))
    rms_i = np.sqrt((err_pt ** 2).mean()).item()
    median_i = float(np.median(err_pt))
    max_i = float(np.max(err_pt))
    n_i = int(err_pt.shape[0])

    total_err2 += float((err_pt ** 2).sum())
    total_pts  += n_i

    per_view_stats.append((good_paths[i], n_i, rms_i, median_i, max_i))

mean_err = np.sqrt(total_err2 / total_pts)

# 排序并打印最差的若干张
per_view_stats_sorted = sorted(per_view_stats, key=lambda x: x[2], reverse=True)
top_k = min(10, len(per_view_stats_sorted))

print("\n===== 每张图的重投影误差（按 RMS 降序，前几张）=====")
for j in range(top_k):
    path_j, n_j, rms_j, med_j, max_j = per_view_stats_sorted[j]
    print(f"[{j+1:02d}] RMS={rms_j:.4f}px  median={med_j:.4f}px  max={max_j:.4f}px  pts={n_j}  file={os.path.basename(path_j)}")

# 导出 CSV 方便筛选/删除离群样本
csv_path = os.path.join(out_dir, "per_view_reprojection_error.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "n_points", "rms_px", "median_px", "max_px"])
    for row in per_view_stats:
        writer.writerow([row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}"])
print(f"[Info] 已导出 per-view 误差: {csv_path}")

# ===== 汇总打印 =====
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print("\n========== 内参结果 ==========")
print(f"图像尺寸:          {w} x {h}")
print(f"RMS（cv2返回）:    {rms:.6f}")
print(f"平均重投影误差px:  {mean_err:.6f}")
print("相机内参矩阵 K：")
print(K)
print(f"fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")





# 保存内参到 YAML
# out_yaml = os.path.join(out_dir, "intrinsics.yaml")
# fs = cv2.FileStorage(out_yaml, cv2.FILE_STORAGE_WRITE)
# fs.write("image_width", w)
# fs.write("image_height", h)
# fs.write("camera_matrix", K)
# fs.write("distortion_coefficients", dist)
# fs.release()
# print(f"[Info] 内参已保存: {out_yaml}")



# # ===== 规则剔除 + 二次标定（可选） =====
# # 规则1：RMS > 0.75 或 最大误差 > 2.0 的样本剔除（你可以改阈值）
# RMS_TH = 0.75
# MAX_TH = 2.0
#
# keep_idx = []
# for i, (path_i, n_i, rms_i, med_i, max_i) in enumerate(per_view_stats):
#     if (rms_i <= RMS_TH) and (max_i <= MAX_TH):
#         keep_idx.append(i)
#
# print(f"\n[Filter] 原始样本 {len(per_view_stats)}，保留 {len(keep_idx)}，剔除 {len(per_view_stats)-len(keep_idx)}")
#
# if len(keep_idx) >= 5:
#     objpoints_f = [objpoints[i] for i in keep_idx]
#     imgpoints_f = [imgpoints[i] for i in keep_idx]
#     paths_f     = [good_paths[i] for i in keep_idx]
#
#     # —— 二次标定（自由解）——
#     rms2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
#         objpoints_f, imgpoints_f, (w, h), None, None
#     )
#
#     # 计算新的平均重投影误差
#     total_err2_b = 0.0
#     total_pts_b = 0
#     for i in range(len(objpoints_f)):
#         reproj, _ = cv2.projectPoints(objpoints_f[i], rvecs2[i], tvecs2[i], K2, dist2)
#         diff = imgpoints_f[i] - reproj
#         err_pt = np.sqrt((diff ** 2).sum(axis=2))
#         total_err2_b += float((err_pt ** 2).sum())
#         total_pts_b  += int(err_pt.shape[0])
#     mean_err2 = np.sqrt(total_err2_b / total_pts_b)
#
#     print("\n========== 二次标定（剔除离群后，自由解）==========")
#     print(f"保留样本数:       {len(keep_idx)}")
#     print(f"RMS（cv2返回）:   {rms2:.6f}")
#     print(f"平均重投影误差:   {mean_err2:.6f}")
#     print("K2 = \n", K2)
#     print(f"fx2={K2[0,0]:.6f}, fy2={K2[1,1]:.6f}, cx2={K2[0,2]:.6f}, cy2={K2[1,2]:.6f}")
#
#     out_yaml = os.path.join(out_dir, "intrinsics.yaml")
#     fs = cv2.FileStorage(out_yaml, cv2.FILE_STORAGE_WRITE)
#     fs.write("image_width", w)
#     fs.write("image_height", h)
#     fs.write("camera_matrix", K2)
#     fs.write("distortion_coefficients", dist2)
#     fs.release()
#     print(f"[Info] 内参已保存: {out_yaml}")
#
#     # —— 三次标定（加入轻约束，常用稳健配置）——
#     # 固定斜切 = 0、切向畸变 = 0（多数镜头足够），固定纵横比（fx≈fy），你也可以去掉 FIX_ASPECT_RATIO 看差别
#     flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO)
#     # 用 K2 作为初值，强制纵横比为 1
#     K_guess = K2.copy()
#     K_guess[0,0] = (K2[0,0] + K2[1,1]) / 2.0
#     K_guess[1,1] = K_guess[0,0]
#     dist_guess = dist2.copy()
#
#     rms3, K3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(
#         objpoints_f, imgpoints_f, (w, h), K_guess, dist_guess, flags=flags
#     )
#
#     # 平均误差
#     total_err3_b = 0.0
#     total_pts3_b = 0
#     for i in range(len(objpoints_f)):
#         reproj, _ = cv2.projectPoints(objpoints_f[i], rvecs3[i], tvecs3[i], K3, dist3)
#         diff = imgpoints_f[i] - reproj
#         err_pt = np.sqrt((diff ** 2).sum(axis=2))
#         total_err3_b += float((err_pt ** 2).sum())
#         total_pts3_b  += int(err_pt.shape[0])
#     mean_err3 = np.sqrt(total_err3_b / total_pts3_b)
#
#     print("\n========== 三次标定（剔除离群 + 轻约束）==========")
#     print(f"RMS（cv2返回）:   {rms3:.6f}")
#     print(f"平均重投影误差:   {mean_err3:.6f}")
#     print("K3 = \n", K3)
#     print(f"fx3={K3[0,0]:.6f}, fy3={K3[1,1]:.6f}, cx3={K3[0,2]:.6f}, cy3={K3[1,2]:.6f}")
#
#     # —— 可选：若主点仍严重跑偏，可再加固定主点在中心的强约束重跑做对比 ——
#     # flags_fix_pp = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
#     # center_K = K3.copy(); center_K[0,2] = w/2; center_K[1,2] = h/2
#     # rms4, K4, dist4, rvecs4, tvecs4 = cv2.calibrateCamera(
#     #     objpoints_f, imgpoints_f, (w, h), center_K, dist3.copy(), flags=flags_fix_pp
#     # )
#     # print("\n[Optional] 固定主点在图像中心后的 K4/RMS：")
#     # print("K4 = \n", K4, "\nRMS=", rms4)
#
# else:
#     print("[Filter] 保留样本太少，跳过二次标定。")
