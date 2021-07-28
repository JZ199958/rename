
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图片
src = cv2.imread('new_color.jpg')
# 图像缩放
result = cv2.resize(src, (45, 50))
#图像旋转
rows, cols, channel = src.shape

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
# 参数：原始图像 旋转参数 元素图像宽高
rotated = cv2.warpAffine(src, M, (cols, rows))
#图像平移
image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

M = np.float32([[1, 0, 0], [0, 1, 100]])
img1 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# 显示图形
titles = ['Source', 'resize','warpAffin','warpAffine']
images = [src, result,rotated,img1]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
