import cv2
import numpy as np





# 01.py
def HistogramEqualization(img):
    # img =src[...,::-1].transpose((0, 2, 3, 1))   # RGB to BGR ,BCHW to  BHWC
    # blue = img[:, :, 0]
    # green = img[:, :, 1]
    # red = img[:, :, 2]
    # img.shape = (3,384,640)
    # request = (640,384,3)
    blue = img[0, :, :]
    green = img[1, :, :]
    red = img[2, :, :]
    blue_equ = cv2.equalizeHist(blue)
    green_equ = cv2.equalizeHist(green)
    red_equ = cv2.equalizeHist(red)
    equ = cv2.merge([blue_equ, green_equ, red_equ])
    equ = equ.transpose((2, 0, 1)) #WHC to CHW

    return equ


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


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)

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
def deHazeDefogging(img):
    # img.shape = (3,384,640)
    # request = (640,384,3)
    out = deHaze(img.transpose((2, 1, 0))/255.0) * 255
    out = out.transpose((2, 1, 0))
    return out


# 03.py
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
def homomorphic_filterDefogging(img):

    out = homomorphic_filter(img.transpose((2, 1, 0)))
    print(out.shape)
    out = out.transpose((2, 1, 0))
    return out

