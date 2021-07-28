import cv2
import matplotlib.pyplot as plt
def histogram(image):
    (row, col) = image.shape
    hist = [0]*256
    for i in range(row):
        for j in range(col):
            hist[image[i,j]] +=1
    return hist
def global_linear_transmation(img):
    maxV=img.max()
    minV=img.min()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ((img[i,j]-minV)*255)/(maxV-minV)
    return img
image0 = cv2.imread("new_color.jpg",0)
plt.figure()
plt.subplot(2,2,1)
plt.imshow(image0,vmin=0, vmax=255,cmap = plt.cm.gray)
plt.title('original image')
image_hist0 = histogram(image0)
plt.subplot(2,2,2)
plt.plot(image_hist0)
image1=global_linear_transmation(image0)
plt.subplot(2,2,3)
plt.imshow(image1,vmin=0, vmax=255,cmap = plt.cm.gray)
image_hist1 = histogram(image1)
plt.subplot(2,2,4)
plt.plot(image_hist1)
plt.show()