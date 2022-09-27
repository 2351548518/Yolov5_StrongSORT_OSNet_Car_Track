import numpy as np
import cv2

width = 852
height = 480


mask_image_temp = np.zeros((height, width), dtype=np.uint8)
# 填充第一个撞线polygon（蓝色）

# 十个点
list_pts_blue = [[184,85],[230,80],[171,315],[447,328]]
ndarray_pts_blue = np.array(list_pts_blue, np.int32)
polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

# 蓝 色盘 b,g,r
blue_color_plate = [255, 0, 0]
# 蓝 polygon图片
blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)



