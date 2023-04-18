import cv2
import numpy as np
import math


# 01.py
def HistogramEqualization(img):
    # img =src[...,::-1].transpose((0, 2, 3, 1))   # RGB to BGR ,BCHW to  BHWC
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    blue_equ = cv2.equalizeHist(blue)
    green_equ = cv2.equalizeHist(green)
    red_equ = cv2.equalizeHist(red)
    out = cv2.merge([red_equ, green_equ, blue_equ])
    # out = out.transpose((2, 0, 1))  # WHC to CHW
    return out


# 02.py
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


def Defog(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    m = m/ 255.0
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - Mask_img) / (1 - Mask_img / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return (Y* 255).astype(np.uint8)


def deHazeDefogging(img):
    # img.shape = (3,384,640)
    # request = (640,384,3)
    out = deHaze(img.transpose((2, 1, 0)) / 255.0) * 255
    out = out.transpose((2, 1, 0))
    return out.astype(np.uint8)


# 03.py
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

    return newf * 255


def HomorphicFilteringDefogging(img):
    # img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    # (678, 1019, 3)
    # img = img.transpose((1, 2, 0))
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    f_d_grayr = HomorphicFiltering(r)
    f_d_grayg = HomorphicFiltering(g)
    f_d_grayb = HomorphicFiltering(b)
    out = cv2.merge([f_d_grayr, f_d_grayg, f_d_grayb]).astype(np.uint8)
    # out = out.transpose((2, 0, 1))  # HWC to CHW
    return out


# 05.py
# 线性拉伸处理
# 去掉最大最小0.5%的像素值 线性拉伸至[0,1]
def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


# 根据半径计算权重参数矩阵
g_para = {}


def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


# 常规的ACE实现
def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh = []
    zw = []
    n = 0
    while n < radius:
        zh.append(0)
        zw.append(0)
        n += 1
    for n in range(height):
        zh.append(n)
    for n in range(width):
        zw.append(n)
    n = 0
    while n < radius:
        zh.append(height - 1)
        zw.append(width - 1)
        n += 1
    # print(zh)
    # print(zw)

    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


# 单通道ACE快速增强实现
def zmIceFast(I, ratio, radius):
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) / 2), int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


# rgb三通道分别增强 ratio是对比度增强因子 radius是卷积模板半径
def zmIceColor(I, ratio=4, radius=3):
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res

def zmIceColorDefog(img):
    res = zmIceColor(img / 255.0) * 255
    return res.astype(np.uint8)