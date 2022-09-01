import cv2
import matplotlib.pyplot as plt

img = cv2.imread("50.jpg")

blue = img[:, :, 0]
green = img[:, :, 1]
red = img[:, :, 2]
blue_equ = cv2.equalizeHist(blue)
green_equ = cv2.equalizeHist(green)
red_equ = cv2.equalizeHist(red)
equ = cv2.merge([blue_equ, green_equ, red_equ])

cv2.imshow("1",img)
cv2.imshow("2",equ)
plt.figure("原始图像直方图")
plt.hist(img.ravel(), 256)
plt.figure("均衡化图像直方图")
plt.hist(equ.ravel(), 256)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
