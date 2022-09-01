import cv2
import matplotlib.pyplot as plt



def HistogramEqualization(src):
    img = src

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    blue_equ = cv2.equalizeHist(blue)
    green_equ = cv2.equalizeHist(green)
    red_equ = cv2.equalizeHist(red)
    equ = cv2.merge([blue_equ, green_equ, red_equ])
    return equ

