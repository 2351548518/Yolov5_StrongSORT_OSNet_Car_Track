# # from https://blog.csdn.net/wsp_1138886114/article/details/95012769
#
import cv2
import numpy as np


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y


if __name__ == '__main__':
    m = deHaze(cv2.imread('fogvideo2.mp4_20221122_224007130.jpg') / 255.0) * 255
    cv2.imwrite('20190708_02.png', m)
# import sys
# import cv2
# import math
# import numpy as np
#
#
# def DarkChannel(im, sz):
#     b, g, r = cv2.split(im)
#     dc = cv2.min(cv2.min(r, g), b)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
#     dark = cv2.erode(dc, kernel)
#     return dark
#
#
# def AtmLight(im, dark):
#     [h, w] = im.shape[:2]
#     imsz = h * w
#     numpx = int(max(math.floor(imsz / 1000), 1))
#     darkvec = dark.reshape(imsz, 1)
#     imvec = im.reshape(imsz, 3)
#
#     indices = darkvec.argsort()
#     indices = indices[imsz - numpx::]
#
#     atmsum = np.zeros([1, 3])
#     for ind in range(1, numpx):
#         atmsum = atmsum + imvec[indices[ind]]
#
#     A = atmsum / numpx;
#     return A
#
#
# def TransmissionEstimate(im, A, sz):
#     omega = 0.95
#     im3 = np.empty(im.shape, im.dtype)
#
#     for ind in range(0, 3):
#         im3[:, :, ind] = im[:, :, ind] / A[0, ind]
#
#     transmission = 1 - omega * DarkChannel(im3, sz)
#     return transmission
#
#
# def Guidedfilter(im, p, r, eps):
#     mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
#     mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
#     mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
#     cov_Ip = mean_Ip - mean_I * mean_p
#
#     mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
#     var_I = mean_II - mean_I * mean_I
#
#     a = cov_Ip / (var_I + eps)
#     b = mean_p - a * mean_I
#
#     mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
#     mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
#
#     q = mean_a * im + mean_b
#     return q
#
#
# def TransmissionRefine(im, et):
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     gray = np.float64(gray) / 255
#     r = 60
#     eps = 0.0001
#     t = Guidedfilter(gray, et, r, eps)
#
#     return t
#
#
# def Recover(im, t, A, tx=0.1):
#     res = np.empty(im.shape, im.dtype)
#     t = cv2.max(t, tx)
#
#     for ind in range(0, 3):
#         res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
#
#     return res
#
#
# if __name__ == '__main__':
#     fn = 'test4.jpg'
#     src = cv2.imread(fn)
#     I = src.astype('float64') / 255
#
#     dark = DarkChannel(I, 15)
#     A = AtmLight(I, dark)
#     te = TransmissionEstimate(I, A, 15)
#     t = TransmissionRefine(src, te)
#     J = Recover(I, t, A, 0.1)
#
#     arr = np.hstack((I, J))
#     cv2.imshow("contrast", arr)
#     cv2.imwrite("car-02-dehaze.png", J * 255)
#     cv2.imwrite("car-02-contrast.png", arr * 255)
#     cv2.waitKey();