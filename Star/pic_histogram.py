import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import tqdm
def analyze_histogram(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    return mean_brightness, std_brightness


# 示例路径（你可以换成你自己的）
image_day1 = glob.glob(r"V:\saved_image\starfold\YDB\250518\*.bmp")[:10]
image_day2 = glob.glob(r"V:\saved_image\starfold\YDB\250507\*.bmp")[:10]
mean1 = []
std1 = []

for i in tqdm.tqdm(image_day1):
    mean, std = analyze_histogram(i)
    hist1.append(hist)
    mean1.append(mean)
    std1.append(std)
    min1.append(min)
    max1.append(max)

hist2 = []
mean2 = []
std2 = []
min2 = []
max2 = []
for i in tqdm.tqdm(image_day2):
    mean, std = analyze_histogram(i)
    hist2.append(hist)
    mean2.append(mean)
    std2.append(std)
    min2.append(min)
    max2.append(max)


# 打印统计信息
print("Day 1 - Mean:", np.mean(mean1), "Std:", np.mean(std1), "Min:", np.mean(min1), "Max:", np.mean(max1))
print("Day 2 - Mean:", np.mean(mean2), "Std:", np.mean(std2), "Min:", np.mean(min2), "Max:", np.mean(max2))

# 可选：可视化两个直方图
plt.plot(np.mean(mean1), label='Day 1')
plt.plot(np.mean(mean2), label='Day 2')
plt.title("Gray Histogram Comparison")
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Pixel Count")
plt.legend()
plt.grid()
plt.show()
