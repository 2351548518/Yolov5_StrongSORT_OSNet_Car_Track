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

        self.device = '0'
        self.weights = "weights/best.pt"
        self.source = " "
        self.imgsz = 640
        # self.conf_thres = 0.25
        # self.max_det = 1000
        # self.view_img=True
        # self.save_txt = False
        # self.save_conf = False
        # self.save_crop = False
        # self.nosave = False
        # self.classes = None
        # self.agnostic_nms = False
        # self.augment = False
        # self.visualize = False
        # self.update = False
        # self.project = ROOT / 'runs/detect'
        # self.name = 'exp'
        # self.exist_ok = False
        # self.line_thickness = 3
        # self.hide_labels = False
        # self.hide_conf = False
        self.half = False

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
        set_logging()
        self.initWeight()

    def initSlots(self):
        self.picButton.clicked.connect(self.button_image_open)
        self.weightButton.clicked.connect(self.button_weight_open)
        self.camButton.clicked.connect(self.button_camera_open)

    def initWeight(self):
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix


        self.pt, self.onnx, self.tflite, self.pb, self.saved_model = (suffix == x for x in suffixes)  # backend booleans
        # 只能使用pt模型
        if self.pt:
            self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights,
                                                                                   map_location=self.device)
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model,
                                                            'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            if classify:  # second-stage classifier
                self.modelc = load_classifier(name='resnet50', n=2)  # initialize
                self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(
                    self.device).eval()
        else:
            print("No weights")

    def initLogo(self):
        pix = QtGui.QPixmap('UI/YOLO.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)
        print("initLogo")

    def button_image_open(self):
        print('button_image_open')
        if self.picButton.text() == "Stop":
            self.stopEvent.set()
            self.camButton.setEnabled(True)
            self.weightButton.setEnabled(True)
            self.picButton.setText("Photo\Video")
        else:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择图片或视频", "", "*.jpg;;*.png;;*.mp4")  # All Files(*)
            if not img_name:
                return
            if img_name.endswith("mp4"):
                self.picButton.setText("Stop")
                self.camButton.setEnabled(False)
                self.weightButton.setEnabled(False)
            thread1 = Thread(target=self.run, kwargs={"weights": self.weights, "source": str(img_name), "nosave": True,
                                                      "view_img": True})
            thread1.start()

    def button_weight_open(self):
        print('button_weight_open')
        weight_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择权重", "", "*.pt")  # All Files(*)
        if not weight_name:
            return
        self.weights = str(weight_name)
        self.initWeight()

    def button_camera_open(self):
        if self.camflag == False:
            print('button_camera_open')
            self.webcam = True
            self.picButton.setEnabled(False)
            self.weightButton.setEnabled(False)
            thread2 = Thread(target=self.run,
                             kwargs={"weights": self.weights, "source": "0", "nosave": True, "view_img": True})
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
        MainWindow.setWindowTitle(_translate("MainWindow", "目标检测Demo"))
        self.picButton.setText(_translate("MainWindow", "Photo\Video"))
        self.camButton.setText(_translate("MainWindow", "Camera"))
        self.weightButton.setText(_translate("MainWindow", "Weights"))

    # 无修改
    def run(
            self,
            # weights=ROOT / 'pretrained/yolov5s.pt',  # model.pt path(s)
            # source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            # imgsz=640,  # inference size (pixels)
            # conf_thres=0.25,  # confidence threshold
            # iou_thres=0.45,  # NMS IOU threshold
            # max_det=1000,  # maximum detections per image
            # device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu

            # view_img=True,  # show results

            # save_txt=False,  # save results to *.txt
            # save_conf=False,  # save confidences in --save-txt labels
            # save_crop=False,  # save cropped prediction boxes
            # nosave=False,  # do not save images/videos
            # classes=None,  # filter by class: --class 0, or --class 0 2 3
            # agnostic_nms=False,  # class-agnostic NMS
            # augment=False,  # augmented inference
            # visualize=False,  # visualize features
            # update=False,  # update all models
            # project=ROOT / 'runs/detect',  # save results to project/name
            # name ='exp',  # save results to project/name
            # exist_ok=False,  # existing project/name ok, do not increment
            # line_thickness=3,  # bounding box thickness (pixels)
            # hide_labels=False,  # hide labels
            # hide_conf=False,  # hide confidences
            # half=False,  # use FP16 half-precision inference
            # dnn=False,  # use OpenCV DNN for ONNX inference


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

        # 无修改
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        #已移动到函数里
        # # Initialize


        # # Load model
        classify = False
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=True)
            bs = len(dataset)  # batch_size
            self.cap = dataset.cap
            self.camflag = True
            self.camButton.setText("Stop")
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=True)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # if pt and device.type != 'cpu':
        #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap in dataset:
            # if self.camflag==False and webcam==True:
            #     return
            t1 = time_sync()
            if self.onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference 只使用pt模型文件


            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(img, augment=augment, visualize=visualize)[0]

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)进行二次分类 未使用
            if classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                names = self.names
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # Add bbox to image 不用判断是否显示if save_img or save_crop or view_img:
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                # print(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                # 主要修改的地方
                self.im0 = annotator.result()
                if view_img:
                    self.result = cv2.cvtColor(self.im0, cv2.COLOR_BGR2BGRA)
                    # self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                              QtGui.QImage.Format_RGB32)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.label.setScaledContents(True)
                    # cv2.imshow(str(p), self.im0)
                    # cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, self.im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, self.im0.shape[1], self.im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(self.im0)

            # 停止检测
            if self.stopEvent.is_set() == True:
                if self.cap.isOpened():
                    self.cap.release()
                self.stopEvent.clear()
                self.initLogo()
                break
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
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
        hide_speed= False, # hide speed
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
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
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
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
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    # 开始检测过程
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
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
                # outputs_prev.append(outputs[i])
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
                            # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else(f'{id} {
                            # conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f} {bbox_speed}'))
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else(f'{id} {conf:.2f}' if hide_class else f'{id} {bbox_speed}'))

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
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
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

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='images/17738409_da3-1-16_Trim.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results', default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--hide-speed', default=False, action='store_true', help='hide speed')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
