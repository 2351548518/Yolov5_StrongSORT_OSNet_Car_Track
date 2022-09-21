import argparse
import math

import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# ImageDefogging 去雾代码
from ImageDefogging.utils import *

# Estimated_speed 测速代码
from EstimatedSpeed.estimatedspeed import Estimated_speed

# PyQT GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import threading
from threading import Thread

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon("UI/icon.png"))
        self.initLogo()
        self.initSlots()

        self.device = ''

        self.yolo_weights = WEIGHTS / 'best.pt'
        self.strong_sort_weights = WEIGHTS / 'osnet_x0_25_market1501.pt'

        self.source = "images/17738409_da3-1-16_Trim.mp4"
        self.imgsz = (640, 640)
        self.half = False
        self.dnn = False

        self.names = None
        self.webcam = False
        self.camflag = False
        self.stopEvent = threading.Event()
        self.cap = None
        self.stride = None
        self.model = None
        self.modelc = None
        self.pt = None
        self.onnx = None
        self.tflite = None
        self.pb = None
        self.saved_model = None

        # Initialize
        self.initWeight()

    # 初始化信号与槽
    def initSlots(self):
        self.picButton.clicked.connect(self.button_image_open)
        self.weightButton.clicked.connect(self.button_weight_open)
        self.camButton.clicked.connect(self.button_camera_open)

    def initWeight(self):
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model 只能使用pt模型
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def initLogo(self):
        pix = QtGui.QPixmap('UI/YOLO.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        if self.picButton.text() == "Stop":
            self.stopEvent.set()
            self.camButton.setEnabled(True)
            self.weightButton.setEnabled(True)
            self.picButton.setText("Video")
        else:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片或视频", "", "*.mp4")  # All Files(*)
            if not img_name:
                return
            if img_name.endswith("mp4"):
                self.picButton.setText("Stop")
                self.camButton.setEnabled(False)
                self.weightButton.setEnabled(False)
            # --yolo - weights
            # weights / best.pt - -strong - sort - weights
            # strong_sort / deep / checkpoint / osnet_x0_25_market1501.pt

            thread1 = Thread(target=self.run,
                             kwargs={"yolo_weights": self.yolo_weights,"strong_sort_weights":self.strong_sort_weights, "source": str(img_name), "nosave": True,
                                     "show_vid": True})
            thread1.start()

    def button_weight_open(self):
        print('button_weight_open')
        weight_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择权重", "", "*.pt")  # All Files(*)
        if not weight_name:
            return
        self.yolo_weights = str(weight_name)
        self.initWeight()

    def button_camera_open(self):
        if self.camflag == False:
            print('button_camera_open')
            self.webcam = True
            self.picButton.setEnabled(False)
            self.weightButton.setEnabled(False)
            thread2 = Thread(target=self.run,
                             kwargs={"yolo_weights": self.yolo_weights, "strong_sort_weights": self.strong_sort_weights,
                                     "source": "0", "nosave": True, "show_vid": True})
            thread2.start()
        else:
            print('button_camera_close')
            self.stopEvent.set()
            self.camflag = False
            self.webcam = False
            self.picButton.setEnabled(True)
            self.weightButton.setEnabled(True)
            self.camButton.setText("Camera")
            self.initLogo()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setFixedSize(900, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(20, -1, 20, -1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.picButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picButton.sizePolicy().hasHeightForWidth())
        self.picButton.setSizePolicy(sizePolicy)
        self.picButton.setMinimumSize(QtCore.QSize(150, 100))
        self.picButton.setMaximumSize(QtCore.QSize(150, 100))
        self.picButton.setSizeIncrement(QtCore.QSize(0, 0))
        self.picButton.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(21)
        self.picButton.setFont(font)
        self.picButton.setObjectName("picButton")
        self.verticalLayout.addWidget(self.picButton)
        self.camButton = QtWidgets.QPushButton(self.centralwidget)
        self.camButton.setMinimumSize(QtCore.QSize(150, 100))
        self.camButton.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(21)
        self.camButton.setFont(font)
        self.camButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.camButton)
        self.weightButton = QtWidgets.QPushButton(self.centralwidget)
        self.weightButton.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.weightButton.sizePolicy().hasHeightForWidth())
        self.weightButton.setSizePolicy(sizePolicy)
        self.weightButton.setMinimumSize(QtCore.QSize(150, 100))
        self.weightButton.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(21)
        self.weightButton.setFont(font)
        self.weightButton.setObjectName("weightButton")
        self.verticalLayout.addWidget(self.weightButton)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 828, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车辆跟踪Demo1"))
        self.picButton.setText(_translate("MainWindow", "Video"))
        self.camButton.setText(_translate("MainWindow", "Camera"))
        self.weightButton.setText(_translate("MainWindow", "Weights"))

    # 无修改
    @torch.no_grad()
    def run(
            self,
            source=ROOT / 'data/images',
            yolo_weights=WEIGHTS / 'weights/best.pt',  # model.pt path(s),
            strong_sort_weights=WEIGHTS / 'osnet_x0_25_market1501.pt',  # model.pt path,
            config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            show_vid=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_vid=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/track',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            hide_speed=False,  # hide speed
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images

        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        # 不能使用wencam
        webcam = self.webcam

        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
        # 无修改
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # 已移动到函数里
        # # Initialize

        # # Load model
        classify = False
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Dataloader
        if webcam:
            # view_img = check_imshow()
            show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=self.pt)
            nr_sources = len(dataset)  # batch_size
            self.cap = dataset.cap
            self.camflag = True
            self.camButton.setText("Stop")
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=True)
            nr_sources = 1  # batch_size
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file(config_strongsort)

        # Create as many strong sort instances as there are video sources
        strongsort_list = []
        for i in range(nr_sources):
            strongsort_list.append(
                StrongSORT(
                    self.strong_sort_weights,
                    self.device,
                    self.half,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

                )
            )
            strongsort_list[i].model.warmup()
        # outputs是输出结果，[i]表示当前输出
        outputs = [None] * nr_sources
        outputs_prev = []

        # Run tracking
        self.model.warmup(imgsz=(1 if self.pt else nr_sources, 3, *imgsz))  # warmup
        # Run inference
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()

            im = deHazeDefogging(im)

            im = torch.from_numpy(im).to(self.device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference 只使用pt模型文件
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process detections 过程检测
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                curr_frames[i] = im0

                txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop
                names = self.names

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                if cfg.STRONGSORT.ECC:  # camera motion compensation 摄像机运动补偿
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size 将框从img_size大小重新缩放到 im0 大小
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to strongsort 将检测传递给strongsort
                    t4 = time_sync()
                    outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # 这里进行测速代码
                    if len(outputs_prev) < 2:
                        outputs_prev.append(outputs[i])
                    else:
                        outputs_prev[:] = [outputs_prev[-1], outputs[i]]

                    # draw boxes for visualization
                    # outputs存放结果 [l,t,w,h,id,cls,]
                    if len(outputs[i]) > 0:
                        # j第几个框，output结果，conf置信度
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                            # 这里是结果
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            bbox_width = output[2] - output[0]
                            fps = 20
                            # if len(outputs_prev) == 2:
                            #     bbox_speed = Estimated_speed(outputs_prev[-2], output, id, fps, bbox_width)
                            # else:
                            #     bbox_speed = "unknown"
                            bbox_speed = Estimated_speed(outputs_prev[-2], output, id, fps, bbox_width)

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                # label = None if hide_labels else (f'{id} {names[c]} ' if hide_conf else (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else (
                                    f'{id} {conf:.2f}' if hide_class else f'{id} {bbox_speed}'))

                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                        c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

                else:
                    strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                # 主要修改的地方
                self.im0 = annotator.result()
                if show_vid:
                    self.result = cv2.cvtColor(self.im0, cv2.COLOR_BGR2BGRA)
                    # self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                              QtGui.QImage.Format_RGB32)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.label.setScaledContents(True)
                    # cv2.imshow(str(p), self.im0)
                    # cv2.waitKey(1)  # 1 millisecond

                if save_vid:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                prev_frames[i] = curr_frames[i]

            # 停止检测
            if self.stopEvent.is_set() == True:
                # cap.isOpened() 判断视频对象是否成功读取，成功读取视频对象返回True。
                if self.cap.isOpened():
                    self.cap.release()
                self.stopEvent.clear()
                self.initLogo()
                break
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
