import numpy as np
import cv2
width = 852
height = 480

# 填充第一个撞线polygon（蓝色）
# 四个点
mask_image_temp = np.zeros((height, width), dtype=np.uint8)
list_pts_blue = [[184,85],[230,80],[171,315],[47,328]]
ndarray_pts_blue = np.array(list_pts_blue, np.int32)
polygon_blue_value_1 = cv2.polylines(mask_image_temp, [ndarray_pts_blue],isClosed=True,  thickness=3,color=1)
polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

# 填充第二个撞线polygon（黄色）
mask_image_temp = np.zeros((height, width), dtype=np.uint8)
# 四个点
list_pts_yellow = [[230,80],[275,83],[298,303],[171,315]]
ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
polygon_yellow_value_2 = cv2.polylines(mask_image_temp, [ndarray_pts_yellow],isClosed=True,  thickness=3,color=2)
polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

# 填充第三个撞线polygon（蓝色）
# 四个点
mask_image_temp = np.zeros((height, width), dtype=np.uint8)
list_pts_blue2 = [[275,83],[328,85],[423,291],[298,303]]
ndarray_pts_blue_2 = np.array(list_pts_blue2, np.int32)
polygon_blue_value_2 = cv2.polylines(mask_image_temp, [ndarray_pts_blue_2],isClosed=True,  thickness=3,color=1)
polygon_blue_value_2 = polygon_blue_value_2[:, :, np.newaxis]

# 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2 + polygon_blue_value_2

# 蓝 色盘 b,g,r
blue_color_plate = [255, 0, 0]
# 蓝 polygon图片
blue_image1 = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
# 蓝 polygon图片
blue_image2 = np.array(polygon_blue_value_2 * blue_color_plate, np.uint8)

# 黄 色盘
yellow_color_plate = [0, 255, 255]
# 黄 polygon图片
yellow_image2 = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

# 彩色图片（值范围 0-255）
color_polygons_image = blue_image1 + yellow_image2 + blue_image2

# list 与蓝色polygon重叠
list_overlapping_blue_polygon = []

# list 与黄色polygon重叠
list_overlapping_yellow_polygon = []

# 下行数量
down_count = 0
# 上行数量
up_count = 0

font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
draw_text_postion = (int((width / 2.0) * 0.01), int((height / 2.0) * 0.05))

# 输出图片
# output_image_frame = cv2.add(output_image_frame, color_polygons_image)

