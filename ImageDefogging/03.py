# # from https://blog.csdn.net/qq_30154571/article/details/109138478
# # https://blog.csdn.net/ab136681/article/details/104518243?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control
#
import cv2
import matplotlib.pyplot as plt
import numpy as np
#
#
# #############        自定义同态滤波器函数      ################r1-亮度
# def homomorphic_filter(src, d0=1, r1=2, rh=2, c=4, h=2.0, l=0.5):
#     # 图像灰度化处理
#     gray = src.copy()
#     if len(src.shape) > 2:  # 维度>2
#         gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#
#     # 图像格式处理
#     gray = np.float64(gray)
#
#     # 设置数据维度n
#     rows, cols = gray.shape
#
#     # 傅里叶变换
#     gray_fft = np.fft.fft2(gray)
#
#     # 将零频点移到频谱的中间，就是中间化处理
#     gray_fftshift = np.fft.fftshift(gray_fft)
#
#     # 生成一个和gray_fftshift一样的全零数据结构
#     dst_fftshift = np.zeros_like(gray_fftshift)
#
#     # arange函数用于创建等差数组，分解f(x,y)=i(x,y)r(x,y)
#     M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))  # 注意，//就是除法
#
#     # 使用频率增强函数处理原函数（也就是处理原图像dst_fftshift）
#     D = np.sqrt(M ** 2 + N ** 2)  # **2是平方
#     Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
#     dst_fftshift = Z * gray_fftshift
#     dst_fftshift = (h - l) * dst_fftshift + l
#
#     # 傅里叶反变换（之前是正变换，现在该反变换变回去了）
#     dst_ifftshift = np.fft.ifftshift(dst_fftshift)
#     dst_ifft = np.fft.ifft2(dst_ifftshift)
#
#     # 选取元素的实部
#     dst = np.real(dst_ifft)
#
#     # dst中，比0小的都会变成0，比0大的都变成255
#     # uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
#     dst = np.uint8(np.clip(dst, 0, 255))
#     return dst
#
#
# ################       主函数开始         ################
#
# img = cv2.imread('pic\\4.jpg', 0)
# # 将图片执行同态滤波器
# img_new = homomorphic_filter(img)
# # cv2.imwrite("TongTai", img_new)
#
# # 输入和输出合并在一起输出
# result = np.hstack((img, img_new))
# # 打印
# cv2.imshow('outputPicName', result)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
#
#
def HomorphicFiltering(original):
    rows, cols = original.shape
    # 对数化区分fi fx numpy的溢出若+1则 255的时候因为是uint 数据上溢 最后1 0000 0000 1被截断所以值为0 log 后为负无穷
    # >> > np.log([1, np.e, np.e ** 2, 0])
    # array([0., 1., 2., -Inf])
    # pyhton 中的溢出，短整形会自动调整为长整型
    # 1 对数化
    original_log = np.log(1e-3 + original)

    # 2 高通滤波
    original_log_F = np.fft.fftshift(np.fft.fft2(original_log))
    # plt.figure()
    # show(np.log(1e-3 + np.abs(original_log_F)), "cc", 1, 1, 1)
    # plt.show()

    HP_Filter = np.zeros(original_log.shape)
    D0 = max(rows, cols)
    for i in range(rows):
        for j in range(cols):
            # rh = 2 rl = 0.2 c = 0.1
            temp = (i - rows / 2) ** 2 + (j - cols / 2) ** 2
            HP_Filter[i, j] = (2 - 0.2) * (1 - np.exp(- 0.1 * temp / (2 * (D0 ** 2)))) + 0.2

    F = np.fft.ifftshift(original_log_F * HP_Filter)

    # 3 反运算得到处理后图像
    f = np.fft.ifft2(F).real
    newf = np.exp(f) - 1
    mi = np.min(newf)
    ma = np.max(newf)
    rang = ma - mi
    for i in range(rows):
        for j in range(cols):
            newf[i, j] = (newf[i, j] - mi) / rang

    return newf


def main():
    f = plt.imread("test4.jpg")
    # r, g, b = cv2.split(f)
    r = f[:, :, 0]
    g = f[:, :, 1]
    b = f[:, :, 2]
    f_d_grayr = HomorphicFiltering(r)
    f_d_grayg = HomorphicFiltering(g)
    f_d_grayb = HomorphicFiltering(b)

    newf = cv2.merge([f_d_grayr, f_d_grayg, f_d_grayb])

    plt.figure("Image")
    plt.imshow(newf)
    plt.show()


main()