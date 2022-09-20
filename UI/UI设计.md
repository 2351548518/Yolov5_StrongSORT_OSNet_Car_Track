# 左侧

## 打开文件

本地文件

摄像机

## 设置

```python
# weights='weights/yolov5s.pt',  # 权重文件地址 默认 weights/best.pt
# source='data/images',          # 测试数据文件(图片或视频)的保存路径 默认data/images
# imgsz=640,                     # 输入图片的大小 默认640(pixels)
# conf_thres=0.25,               # object置信度阈值 默认0.25  用在nms中
# iou_thres=0.45,                # 做nms的iou阈值 默认0.45   用在nms中
# max_det=1000,                  # 每张图片最多的目标数量  用在nms中
# device='',                     # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
# view_img=False,                # 是否展示预测之后的图片或视频 默认False
# save_txt=False,   # 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
# save_conf=False,  # 是否保存预测每个目标的置信度到预测tx文件中 默认True
# save_crop=False,  # 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
# nosave=False,     # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
# classes=None,     # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
# agnostic_nms=False,     # 进行nms是否也除去不同类别之间的框 默认False
# augment=False,          # 预测是否也要采用数据增强 TTA 默认False
# update=False,           # 是否将optimizer从ckpt中删除  更新模型  默认False
# project='runs/detect',  # 当前测试结果放在哪个主文件夹下 默认runs/detect
# name='exp',             # 当前测试结果放在run/detect下的文件名  默认是exp  =>  run/detect/exp
# exist_ok=False,         # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
# line_thickness=3,       # bounding box thickness (pixels)   画框的框框的线宽  默认是 3
# hide_labels=False,      # 画出的框框是否需要隐藏label信息 默认False
# hide_conf=False,        # 画出的框框是否需要隐藏conf信息 默认False
# half=False,             # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
# prune_model=False,      # 是否使用模型剪枝 进行推理加速
# fuse=False,             # 是否使用conv + bn融合技术 进行推理加速
```

设置界面，整个设置界面放在一个滚动条中

```
下拉选择框
yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
下拉选择框
strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
下拉选择框
config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
文本输入框
imgsz=(640, 640),  # inference size (height, width)
滑动条或者文本输入框
conf_thres=0.25,  # confidence threshold
滑动条或者文本输入框
iou_thres=0.45,  # NMS IOU threshold
文本输入框
max_det=1000,  # maximum detections per image
下拉选择框
device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
单选框
show_vid=False,  # show results
单选框
save_txt=False,  # save results to *.txt
单选框
save_conf=False,  # save confidences in --save-txt labels
单选框
save_crop=False,  # save cropped prediction boxes
单选框
save_vid=False,  # save confidences in --save-txt labels
单选框
nosave=False,  # do not save images/videos
复选框
classes=None,  # filter by class: --class 0, or --class 0 2 3
单选框
agnostic_nms=False,  # class-agnostic NMS
单选框
augment=False,  # augmented inference
单选框
visualize=False,  # visualize features
单选框
update=False,  # update all models
弹窗选择
project=ROOT / 'runs/track',  # save results to project/name
文本输入框
name='exp',  # save results to project/name
单选框
exist_ok=False,  # existing project/name ok, do not increment
文本输入框
line_thickness=3,  # bounding box thickness (pixels)
单选框
hide_labels=False,  # hide labels
单选框
hide_conf=False,  # hide confidences
单选框
hide_class=False,  # hide IDs
单选框
hide_speed= False, # hide speed
单选框
half=False,  # use FP16 half-precision inference
单选框
dnn=False,  # use OpenCV DNN for ONNX inference
```

## 开始跟踪

# 右侧

两个视频窗口，一个放置原视频或者去雾后的视频，一个放跟踪后的视频

底部放一个多行文本框，用来输出结果 ，右侧放置一个截屏按钮，可以获取当前识别到的信息



把上方的菜单栏删除

自己写

![image-20220919000227969](https://raw.githubusercontent.com/2351548518/images/main/20220717/202209190003564.png)

## 结果展示设置

可以将原视频和处理后的视频合成一个视频（每一帧的图片结果进行拼接）然后进行展示（一个展示窗口就行）

这样就规避了线程问题，

应该可以解决问题。

![image-20220920001104417](https://raw.githubusercontent.com/2351548518/images/main/20220717/202209200011514.png)
