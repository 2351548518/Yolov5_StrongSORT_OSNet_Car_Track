# from https://blog.csdn.net/qq_30154571/article/details/109138478

import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np


#############        自定义同态滤波器函数      ################r1-亮度
def homomorphic_filter(src, d0=1, r1=2, rh=2, c=4, h=2.0, l=0.5):
    # 图像灰度化处理
    gray = src.copy()
    if len(src.shape) > 2:  # 维度>2
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 图像格式处理
    gray = np.float64(gray)

    # 设置数据维度n
    rows, cols = gray.shape

    # 傅里叶变换
    gray_fft = np.fft.fft2(gray)

    # 将零频点移到频谱的中间，就是中间化处理
    gray_fftshift = np.fft.fftshift(gray_fft)

    # 生成一个和gray_fftshift一样的全零数据结构
    dst_fftshift = np.zeros_like(gray_fftshift)

    # arange函数用于创建等差数组，分解f(x,y)=i(x,y)r(x,y)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))  # 注意，//就是除法

    # 使用频率增强函数处理原函数（也就是处理原图像dst_fftshift）
    D = np.sqrt(M ** 2 + N ** 2)  # **2是平方
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l

    # 傅里叶反变换（之前是正变换，现在该反变换变回去了）
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)

    # 选取元素的实部
    dst = np.real(dst_ifft)

    # dst中，比0小的都会变成0，比0大的都变成255
    # uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


################       主函数开始         ################

img = cv2.imread('pic\\4.jpg', 0)
# 将图片执行同态滤波器
img_new = homomorphic_filter(img)
# cv2.imwrite("TongTai", img_new)

# 输入和输出合并在一起输出
result = np.hstack((img, img_new))
# 打印
cv2.imshow('outputPicName', result)

cv2.waitKey()
cv2.destroyAllWindows()



