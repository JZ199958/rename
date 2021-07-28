# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread('new_color.jpg')
source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, chn = img.shape

# 加噪声
for i in range(5000):
    x = np.random.randint(0, rows)
    y = np.random.randint(0, cols)
    img[x, y, :] = 255

# 均值滤波
result = cv2.blur(source, (5, 5))
result1 = cv2.medianBlur(img, 3)
# 显示图形
titles = ['Source Image', 'noise','Blur Image','medianBlur']
images = [source, img,result,result1]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

