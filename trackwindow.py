# -*- coding: utf-8 -*-
import os

from PyQt5.QtCore import QObject, pyqtSignal

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

# # CountCar 计数代码
# from CountCar.CountCar import *

# 界面实现 PyQT GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import threading
from threading import Thread

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
class Ui_MainWindow(QtWidgets.QMainWindow):
    sendmsg = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon("UI/icon.png"))
        self.initLogo()

        # 去雾开关：单选框
        self.DefogOpen = False

        # 设置默认值
        # 下拉选择框(文件打开按钮) yolo权重
        self.yolo_weights = WEIGHTS / 'best.pt'  # model.pt path(s),
        # 下拉选择框（文件打开按钮）deepsort权重
        self.strong_sort_weights = WEIGHTS / 'osnet_x0_25_market1501.pt'  # model.pt path,

        # 增加
        self.source = "images/17738409_da3-1-16_Trim.mp4"
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
        # 导出图片使用
        self.save_image_flag = False
        # 记录视频使用
        self.save_video_flag = False

        # # 下拉选择框（文件打开按钮）config_strongsort
        # self.config_strongsort = ROOT / 'strong_sort/configs/strong_sort.yaml',
        # 文本输入框（或者不设置）
        self.project = ROOT / 'runs/track'  # save results to project/name
        # 文本输入框（或者不设置）
        self.name = "exp"  # save results to project/name

        # 限制输入只能是数字

        # 文本输入框 ：输入图片的大小
        self.imgsz = (640, 640)  # inference size (height, width)
        # 滑动条或者文本输入框 ： 置信度阈值：浮点校验器 [0，1]，精度：小数点后2位
        self.conf_thres = 0.25  # confidence threshold
        # 滑动条或者文本输入框 ： nms的iou阈值 ：浮点校验器 [0，1]，精度：小数点后2位
        self.iou_thres = 0.45  # NMS IOU threshold
        # 文本输入框 ： 图片最多目标数量 ：整数校验器 （1，1000）
        self.max_det = 1000  # maximum detections per image
        # 文本输入框 ： 框线宽度（pixels）： 整数校验器 （1，5）
        self.line_thickness = 1  # bounding box thickness (pixels)
        # 下拉选择框 ：
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        # 多选框
        self.show_vid = True  # show results ： 结果展示
        self.save_txt = False  # save results to *.txt 坐标保存
        self.save_conf = False  # save confidences in --save-txt labels ： 置信度保存
        self.save_crop = False  # save cropped prediction boxes ： 目标保存
        self.save_vid = False  # save confidences in --save-txt labels ： 预测结果保存

        self.nosave = False  # do not save images/videos ： 不保存预测结果
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models

        self.exist_ok = False  # existing project/name ok, do not increment
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.hide_class = False  # hide IDs
        self.hide_speed = False  # hide speed

        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        # 文本输入框
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        # 去雾开关信号与槽绑定
        self.initRadioSlots()
        # # 按钮信号与槽绑定
        self.initBtnSlots()
        # # 单行文本框信号与槽绑定
        # self.initLineEditSlots()
        # # 复选框信号与槽绑定
        # self.initCheckBoxSlots()

        # 参数初始设置
        self.initParameter()
        # Initialize
        self.initWeight()
        # 实例化状态栏
        self.statusBar = QtWidgets.QStatusBar()
        # 设置状态栏，类似布局设置
        self.setStatusBar(self.statusBar)

    # 参数初始设置
    def initParameter(self):
        # 去雾开关
        self.radioButtonDefogOpen.setChecked(self.DefogOpen)
        # 文本输入框
        self.ProjectLineEdit.setText(str(self.project))
        self.NamelineEdit.setText(self.name)
        self.imgszlineEdit.setText(str(self.imgsz))
        self.conf_threslineEdit.setText(str(self.conf_thres))
        self.iou_threslineEdit.setText(str(self.iou_thres))
        self.max_detlineEdit.setText(str(self.max_det))
        self.line_thicknesslineEdit.setText(str(self.line_thickness))

        # 复选框
        self.checkBoxshow_vid.setChecked(self.show_vid)
        self.checkBoxsave_txt.setChecked(self.save_txt)
        self.checkBoxsave_conf.setChecked(self.save_conf)
        self.checkBoxsave_crop.setChecked(self.save_crop)
        self.checkBoxsave_vid.setChecked(self.save_vid)

        self.checkBoxnosave.setChecked(self.nosave)
        self.checkBoxagnostic_nms.setChecked(self.agnostic_nms)
        self.checkBoxaugment.setChecked(self.augment)
        self.checkBoxvisualize.setChecked(self.visualize)
        self.checkBoxupdate.setChecked(self.update)

        self.checkBoxexist_ok.setChecked(self.exist_ok)
        self.checkBoxhide_labels.setChecked(self.hide_labels)
        self.checkBoxhide_conf.setChecked(self.hide_conf)
        self.checkBoxhide_class.setChecked(self.hide_class)
        self.checkBoxhide_speed.setChecked(self.hide_speed)

        self.checkBoxhalf.setChecked(self.half)
        self.checkBoxdnn.setChecked(self.dnn)

    def Append(self, msg):
        self.sendmsg.emit(msg)

    def setappendPlainText(self, msg):
        self.textEditShowResult.appendPlainText(msg)

    # 按钮信号与槽绑定
    def initBtnSlots(self):
        # 打开视频按钮
        self.VideoOpenBtn.clicked.connect(self.btn_VideoOpen)
        # 打开摄像机按钮
        self.CameraOpenBtn.clicked.connect(self.btn_CameraOpen)
        # yolo权重按钮
        self.YoloWeightsBtn.clicked.connect(self.btn_YoloWeights)
        # deepsort权重按钮
        self.StrongsortWeightsBtn.clicked.connect(self.btn_StrongsortWeights)
        # # config_strongsort按钮
        # self.ConfigStrongsortBtn.clicked.connect(self.btn_ConfigStrongsort)
        # # 开始跟踪按钮
        # self.StartTrackBtn.clicked.connect(self.btn_StartTrack)
        # 导出当前\n视频帧图片按钮
        self.OutputSaveBtn.clicked.connect(self.btn_OutputSave)
        # 记录视频按钮
        self.OutputVideoSaveBtn.clicked.connect(self.btn_OutputVideoSave)
        # 信息展示文本框
        self.sendmsg.connect(self.setappendPlainText)
        # 参数更新按钮
        self.pushButtonParameterSet.clicked.connect(self.initCheckBoxSet)

    # 去雾开关信号与槽绑定
    def initRadioSlots(self):
        self.radioButtonDefogOpen.toggled.connect(self.rbtn_DefogOpen)

    # 文本输入框信号与槽绑定
    def initLineEditSlots(self):
        # 测试结果放置文件夹
        self.ProjectLineEdit.editingFinished.connect(self.ledit_ProjectLineEdit)
        # 测试结果文件夹名称
        self.NamelineEdit.editingFinished.connect(self.ledit_NamelineEdit)
        # 输入图片的大小
        self.imgszlineEdit.editingFinished.connect(self.ledit_imgszlineEdit)
        # 置信度阈值
        self.conf_threslineEdit.editingFinished.connect(self.ledit_conf_threslineEdit)
        # nms的iou阈值
        self.iou_threslineEdit.editingFinished.connect(self.ledit_iou_threslineEdit)
        # 图片最多目标数量
        self.max_detlineEdit.editingFinished.connect(self.ledit_max_detlineEdit)
        # 框线宽度（pixels）
        self.line_thicknesslineEdit.editingFinished.connect(self.ledit_line_thicknesslineEdit)

    def initCheckBoxSet(self):
        # 文本输入框
        self.project = self.ProjectLineEdit.text()
        self.name = self.NamelineEdit.text()
        self.imgsz = eval(self.imgszlineEdit.text())
        self.conf_thres = eval(self.conf_threslineEdit.text())
        self.iou_thres = eval(self.iou_threslineEdit.text())
        self.max_det = eval(self.max_detlineEdit.text())
        self.line_thickness = eval(self.line_thicknesslineEdit.text())

        # 复选框
        self.show_vid = self.checkBoxshow_vid.isChecked()
        self.save_txt = self.checkBoxsave_txt.isChecked()
        self.save_conf = self.checkBoxsave_conf.isChecked()
        self.save_crop = self.checkBoxsave_crop.isChecked()
        self.save_vid = self.checkBoxsave_vid.isChecked()

        self.nosave = self.checkBoxnosave.isChecked()
        self.agnostic_nms = self.checkBoxagnostic_nms.isChecked()
        self.augment = self.checkBoxaugment.isChecked()
        self.visualize = self.checkBoxvisualize.isChecked()
        self.update = self.checkBoxupdate.isChecked()

        self.exist_ok = self.checkBoxexist_ok.isChecked()
        self.hide_labels = self.checkBoxhide_labels.isChecked()
        self.hide_conf = self.checkBoxhide_conf.isChecked()
        self.hide_class = self.checkBoxhide_class.isChecked()
        self.hide_speed = self.checkBoxhide_speed.isChecked()

        self.half = self.checkBoxhalf.isChecked()
        self.dnn = self.checkBoxdnn.isChecked()

    def initWeight(self):
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model 只能使用pt模型
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def initLogo(self):
        pix = QtGui.QPixmap('UI/YOLO.png')
        self.VideoShowLabel.setScaledContents(True)
        self.VideoShowLabel.setPixmap(pix)

    def rbtn_DefogOpen(self):
        if self.radioButtonDefogOpen.isChecked():
            self.DefogOpen = True
        else:
            self.DefogOpen = False

    def btn_VideoOpen(self):
        print('btn_VideoOpen_open')
        if self.VideoOpenBtn.text() == "停止":
            self.stopEvent.set()
            self.CameraOpenBtn.setEnabled(True)
            self.YoloWeightsBtn.setEnabled(True)
            self.StrongsortWeightsBtn.setEnabled(True)
            self.VideoOpenBtn.setText("打开视频")
            # self.initLogo()

        else:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片或视频", "", "*.mp4")  # All Files(*)
            if not img_name:
                return
            if img_name.endswith("mp4"):
                self.VideoOpenBtn.setText("停止")
                self.CameraOpenBtn.setEnabled(False)
                self.YoloWeightsBtn.setEnabled(False)
                self.StrongsortWeightsBtn.setEnabled(False)
            thread1 = Thread(target=self.run,
                             kwargs={"yolo_weights": self.yolo_weights,
                                     "strong_sort_weights": self.strong_sort_weights,
                                     "source": str(img_name),
                                     "nosave": self.nosave,
                                     "conf_thres": self.conf_thres,
                                     "iou_thres": self.iou_thres,
                                     "max_det": self.max_det,
                                     "device":self.device,
                                     "show_vid": self.show_vid,
                                     "save_txt": self.save_txt,
                                     "save_conf": self.save_conf,
                                     "save_crop": self.save_crop,
                                     "save_vid": self.save_vid,
                                     "classes": self.classes,
                                     "agnostic_nms": self.agnostic_nms,
                                     "augment": self.augment,
                                     "visualize": self.visualize,
                                     "update": self.update,
                                     "project": self.project,
                                     "name": self.name,
                                     "exist_ok": self.exist_ok,
                                     "line_thickness": self.line_thickness,
                                     "hide_labels": self.hide_labels,
                                     "hide_conf": self.hide_conf,
                                     "hide_class": self.hide_class,
                                     "hide_speed": self.hide_speed,
                                     "half": self.half,
                                     })
            thread1.start()

    def btn_CameraOpen(self):
        if self.camflag == False:
            print('button_camera_open')
            self.webcam = True
            self.VideoOpenBtn.setEnabled(False)
            self.YoloWeightsBtn.setEnabled(False)
            self.StrongsortWeightsBtn.setEnabled(False)
            thread2 = Thread(target=self.run,
                             kwargs={"yolo_weights": self.yolo_weights,
                                     "strong_sort_weights": self.strong_sort_weights,
                                     "imgsz": self.imgsz,
                                     "source": "0",
                                     "conf_thres": self.conf_thres,
                                     "iou_thres": self.iou_thres,
                                     "max_det": self.max_det,
                                     "show_vid": self.show_vid,
                                     "save_txt": self.save_txt,
                                     "save_conf": self.save_conf,
                                     "save_crop": self.save_crop,
                                     "save_vid": self.save_vid,
                                     "nosave": self.nosave,
                                     "classes": self.classes,
                                     "agnostic_nms": self.agnostic_nms,
                                     "augment": self.augment,
                                     "visualize": self.visualize,
                                     "update": self.update,
                                     "project": self.project,
                                     "name": self.name,
                                     "exist_ok": self.exist_ok,
                                     "line_thickness": self.line_thickness,
                                     "hide_labels": self.hide_labels,
                                     "hide_conf": self.hide_conf,
                                     "hide_class": self.hide_class,
                                     "hide_speed": self.hide_speed,
                                     "half": self.half,
                                     })
            thread2.start()
        else:
            print('button_camera_close')
            self.stopEvent.set()
            self.camflag = False
            self.webcam = False
            self.VideoOpenBtn.setEnabled(True)
            self.YoloWeightsBtn.setEnabled(True)
            self.StrongsortWeightsBtn.setEnabled(True)
            self.CameraOpenBtn.setText("打开摄像机")
            # self.initLogo()

    def btn_YoloWeights(self):
        print('btn_YoloWeights_open')
        weight_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择Yolo权重", "", "*.pt")  # All Files(*)
        if not weight_name:
            return
        self.yolo_weights = str(weight_name)
        self.initWeight()

    def btn_StrongsortWeights(self):
        print('btn_StrongsortWeights_open')
        weight_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择strong_sort权重", "", "*.pt")  # All Files(*)
        if not weight_name:
            return
        self.strong_sort_weights = str(weight_name)
        # self.initWeight()

    # def btn_StartTrack(self):
    #     pass

    def btn_OutputVideoSave(self):
        print('btn_OutputVideoSave_open')
        if not self.save_video_flag:
            self.save_video_flag = True
            self.OutputVideoSaveBtn.setText("停止")
        else:
            self.save_video_flag = False
            self.OutputVideoSaveBtn.setText("开始记录\n结果")

    def btn_OutputSave(self):
        print('btn_OutputSave_open')
        if not self.save_image_flag:
            self.save_image_flag = True

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)
        # 最外层布局
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 给主界面设置水平布局
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        # 左侧垂直布局
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        # 打开视频按钮
        self.VideoOpenBtn = QtWidgets.QPushButton(self.centralwidget)
        self.VideoOpenBtn.setMinimumSize(QtCore.QSize(200, 0))
        self.VideoOpenBtn.setObjectName("VideoOpenBtn")
        self.verticalLayout.addWidget(self.VideoOpenBtn)
        # 打开摄像机按钮
        self.CameraOpenBtn = QtWidgets.QPushButton(self.centralwidget)
        self.CameraOpenBtn.setObjectName("CameraOpenBtn")
        self.verticalLayout.addWidget(self.CameraOpenBtn)
        # 分割线
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        # 去雾开关单选框
        self.radioButtonDefogOpen = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonDefogOpen.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButtonDefogOpen.setObjectName("radioButtonDefogOpen")
        self.verticalLayout.addWidget(self.radioButtonDefogOpen)
        # 分割线2
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        # 滚动条，用来设置繁多的参数
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 181, 1000))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(0, 1000))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        # 设置滚动条内的垂直布局
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        # 第一个groupbox 用来放置权重设置
        self.groupBox = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        # 给第一个groupbox设置垂直布局
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        # yolo权重设置按钮
        self.YoloWeightsBtn = QtWidgets.QPushButton(self.groupBox)
        self.YoloWeightsBtn.setObjectName("YoloWeightsBtn")
        self.verticalLayout_6.addWidget(self.YoloWeightsBtn)
        # deepsort权重设置按钮
        self.StrongsortWeightsBtn = QtWidgets.QPushButton(self.groupBox)
        self.StrongsortWeightsBtn.setObjectName("StrongsortWeightsBtn")
        self.verticalLayout_6.addWidget(self.StrongsortWeightsBtn)
        # # ConfigStrongsor设置按钮
        # self.ConfigStrongsortBtn = QtWidgets.QPushButton(self.groupBox)
        # self.ConfigStrongsortBtn.setObjectName("ConfigStrongsortBtn")
        # self.verticalLayout_6.addWidget(self.ConfigStrongsortBtn)
        # 设置垂直布局
        self.verticalLayout_3.addWidget(self.groupBox)
        # 第二个groupbox 用来设置测试结果的存放目录
        self.groupBox_2 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        # 给groupbox设置垂直布局
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        # 提示标签 测试结果放置文件夹
        self.labelproject = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelproject.sizePolicy().hasHeightForWidth())
        self.labelproject.setSizePolicy(sizePolicy)
        self.labelproject.setObjectName("labelproject")
        self.verticalLayout_8.addWidget(self.labelproject)
        # 测试结果放置文件夹目录设置
        self.ProjectLineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.ProjectLineEdit.setPlaceholderText("runs/track")
        self.ProjectLineEdit.setObjectName("ProjectLineEdit")
        self.verticalLayout_8.addWidget(self.ProjectLineEdit)
        # 提示标签 测试结果文件夹名称
        self.labelname = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelname.sizePolicy().hasHeightForWidth())
        self.labelname.setSizePolicy(sizePolicy)
        self.labelname.setObjectName("labelname")
        self.verticalLayout_8.addWidget(self.labelname)
        # 测试结果文件夹名称 设置
        self.NamelineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.NamelineEdit.setPlaceholderText("")
        self.NamelineEdit.setObjectName("NamelineEdit")
        self.verticalLayout_8.addWidget(self.NamelineEdit)

        self.verticalLayout_3.addWidget(self.groupBox_2)
        # 第三个groupbox 用来存放参数设置
        self.groupBox_3 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        # label 输入图片的大小
        self.labelimgsz = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelimgsz.sizePolicy().hasHeightForWidth())
        self.labelimgsz.setSizePolicy(sizePolicy)
        self.labelimgsz.setObjectName("labelimgsz")
        self.verticalLayout_5.addWidget(self.labelimgsz)
        # 输入图片的大小 文本输入框
        self.imgszlineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.imgszlineEdit.setObjectName("imgszlineEdit")
        self.verticalLayout_5.addWidget(self.imgszlineEdit)
        # label 置信度阈值
        self.labelconf_thres = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelconf_thres.sizePolicy().hasHeightForWidth())
        self.labelconf_thres.setSizePolicy(sizePolicy)
        self.labelconf_thres.setObjectName("labelconf_thres")
        self.verticalLayout_5.addWidget(self.labelconf_thres)
        # 置信度阈值 浮点校验器 [0，1]，精度：小数点后2位
        doubleValidator_conf_threslineEdit = QDoubleValidator(self)
        doubleValidator_conf_threslineEdit.setRange(0, 1)
        doubleValidator_conf_threslineEdit.setNotation(QDoubleValidator.StandardNotation)
        doubleValidator_conf_threslineEdit.setDecimals(2)
        # 置信度阈值 文本输入框
        self.conf_threslineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.conf_threslineEdit.setObjectName("conf_threslineEdit")
        # 置信度阈值 设置校验器
        self.conf_threslineEdit.setValidator(doubleValidator_conf_threslineEdit)

        self.verticalLayout_5.addWidget(self.conf_threslineEdit)
        # label nms的iou阈值
        self.labeliou_thres = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labeliou_thres.sizePolicy().hasHeightForWidth())
        # nms的iou阈值 ：浮点校验器[0，1]，精度：小数点后2位
        doubleValidator_labeliou_thres = QDoubleValidator(self)
        doubleValidator_labeliou_thres.setRange(0, 1)
        doubleValidator_labeliou_thres.setNotation(QDoubleValidator.StandardNotation)
        doubleValidator_labeliou_thres.setDecimals(2)
        self.labeliou_thres.setSizePolicy(sizePolicy)
        self.labeliou_thres.setObjectName("labeliou_thres")
        self.verticalLayout_5.addWidget(self.labeliou_thres)
        # nms的iou阈值 文本输入框
        self.iou_threslineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.iou_threslineEdit.setObjectName("iou_threslineEdit")
        self.verticalLayout_5.addWidget(self.iou_threslineEdit)
        # label 图片最多目标数量
        self.labelmax_det = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelmax_det.sizePolicy().hasHeightForWidth())
        self.labelmax_det.setSizePolicy(sizePolicy)
        self.labelmax_det.setObjectName("labelmax_det")
        self.verticalLayout_5.addWidget(self.labelmax_det)
        # 图片最多目标数量 文本输入框
        self.max_detlineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.max_detlineEdit.setObjectName("max_detlineEdit")
        # 文本输入框 ： 图片最多目标数量 ：整数校验器 （1，1000）
        intValidator_max_detlineEdit = QIntValidator(self)
        intValidator_max_detlineEdit.setRange(1, 1000)
        self.max_detlineEdit.setValidator(intValidator_max_detlineEdit)

        self.verticalLayout_5.addWidget(self.max_detlineEdit)
        # label 框线宽度（pixels）
        self.labelline_thickness = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelline_thickness.sizePolicy().hasHeightForWidth())
        self.labelline_thickness.setSizePolicy(sizePolicy)
        self.labelline_thickness.setObjectName("labelline_thickness")
        self.verticalLayout_5.addWidget(self.labelline_thickness)
        # 框线宽度（pixels） 文本输入框
        self.line_thicknesslineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.line_thicknesslineEdit.setObjectName("line_thicknesslineEdit")
        # 文本输入框 ： 框线宽度（pixels）： 整数校验器 （1，9）
        intValidator_line_thicknesslineEdit = QIntValidator(self)
        intValidator_line_thicknesslineEdit.setRange(1, 9)
        self.line_thicknesslineEdit.setValidator(intValidator_line_thicknesslineEdit)
        self.verticalLayout_5.addWidget(self.line_thicknesslineEdit)

        self.verticalLayout_3.addWidget(self.groupBox_3)
        # 第四个groupbox 用来设置功能开关
        self.groupBox_4 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setObjectName("groupBox_4")
        # 设置垂直布局
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        # 各种功能设置
        self.checkBoxshow_vid = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxshow_vid.setObjectName("checkBoxshow_vid")
        self.verticalLayout_4.addWidget(self.checkBoxshow_vid)
        self.checkBoxsave_txt = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxsave_txt.setObjectName("checkBoxsave_txt")
        self.verticalLayout_4.addWidget(self.checkBoxsave_txt)
        self.checkBoxsave_conf = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxsave_conf.setObjectName("checkBoxsave_conf")
        self.verticalLayout_4.addWidget(self.checkBoxsave_conf)
        self.checkBoxsave_crop = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxsave_crop.setObjectName("checkBoxsave_crop")
        self.verticalLayout_4.addWidget(self.checkBoxsave_crop)
        self.checkBoxsave_vid = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxsave_vid.setObjectName("checkBoxsave_vid")
        self.verticalLayout_4.addWidget(self.checkBoxsave_vid)
        self.checkBoxnosave = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxnosave.setObjectName("checkBoxnosave")
        self.verticalLayout_4.addWidget(self.checkBoxnosave)
        self.checkBoxagnostic_nms = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxagnostic_nms.setObjectName("checkBoxagnostic_nms")
        self.verticalLayout_4.addWidget(self.checkBoxagnostic_nms)
        self.checkBoxaugment = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxaugment.setObjectName("checkBoxaugment")
        self.verticalLayout_4.addWidget(self.checkBoxaugment)
        self.checkBoxvisualize = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxvisualize.setObjectName("checkBoxvisualize")
        self.verticalLayout_4.addWidget(self.checkBoxvisualize)
        self.checkBoxupdate = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxupdate.setObjectName("checkBoxupdate")
        self.verticalLayout_4.addWidget(self.checkBoxupdate)
        self.checkBoxexist_ok = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxexist_ok.setObjectName("checkBoxexist_ok")
        self.verticalLayout_4.addWidget(self.checkBoxexist_ok)
        self.checkBoxhide_labels = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxhide_labels.setObjectName("checkBoxhide_labels")
        self.verticalLayout_4.addWidget(self.checkBoxhide_labels)
        self.checkBoxhide_conf = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxhide_conf.setObjectName("checkBoxhide_conf")
        self.verticalLayout_4.addWidget(self.checkBoxhide_conf)
        self.checkBoxhide_class = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxhide_class.setObjectName("checkBoxhide_class")
        self.verticalLayout_4.addWidget(self.checkBoxhide_class)
        self.checkBoxhide_speed = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxhide_speed.setObjectName("checkBoxhide_speed")
        self.verticalLayout_4.addWidget(self.checkBoxhide_speed)
        self.checkBoxhalf = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxhalf.setObjectName("checkBoxhalf")
        self.verticalLayout_4.addWidget(self.checkBoxhalf)
        self.checkBoxdnn = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBoxdnn.setObjectName("checkBoxdnn")
        self.verticalLayout_4.addWidget(self.checkBoxdnn)

        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)

        self.pushButtonParameterSet = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButtonParameterSet.sizePolicy().hasHeightForWidth())
        self.pushButtonParameterSet.setSizePolicy(sizePolicy)
        self.pushButtonParameterSet.setObjectName("pushButtonParameterSet")
        self.verticalLayout.addWidget(self.pushButtonParameterSet)

        self.horizontalLayout_3.addLayout(self.verticalLayout)
        # 分割线
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.horizontalLayout_3.addWidget(self.line_7)
        # 右侧垂直布局
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        # label 结果展示
        self.labelshowresult = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelshowresult.sizePolicy().hasHeightForWidth())
        self.labelshowresult.setSizePolicy(sizePolicy)
        self.labelshowresult.setAlignment(QtCore.Qt.AlignCenter)
        self.labelshowresult.setObjectName("labelshowresult")
        self.verticalLayout_2.addWidget(self.labelshowresult)
        # 分割线
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_2.addWidget(self.line_6)
        # 用来播放视频的label
        self.VideoShowLabel = QtWidgets.QLabel(self.centralwidget)
        self.VideoShowLabel.setMinimumSize(QtCore.QSize(640, 400))
        self.VideoShowLabel.setText("")
        # self.VideoShowLabel.setScaledContents(True)
        self.VideoShowLabel.setObjectName("VideoShowLabel")
        self.verticalLayout_2.addWidget(self.VideoShowLabel)
        # 分割线
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_2.addWidget(self.line_4)
        # 下侧水平布局
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        # 结果展示textEdit
        self.textEditShowResult = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.textEditShowResult.setObjectName("textEditShowResult")
        self.horizontalLayout_4.addWidget(self.textEditShowResult)
        # 分割线
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout_4.addWidget(self.line_5)
        # 导出当前视频帧图片按钮
        self.OutputVideoSaveBtn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OutputVideoSaveBtn.sizePolicy().hasHeightForWidth())
        self.OutputVideoSaveBtn.setSizePolicy(sizePolicy)
        self.OutputVideoSaveBtn.setObjectName("OutputVideoSaveBtn")
        self.horizontalLayout_4.addWidget(self.OutputVideoSaveBtn)

        # 导出当前视频帧图片按钮
        self.OutputSaveBtn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OutputSaveBtn.sizePolicy().hasHeightForWidth())
        self.OutputSaveBtn.setSizePolicy(sizePolicy)
        self.OutputSaveBtn.setObjectName("OutputSaveBtn")
        self.horizontalLayout_4.addWidget(self.OutputSaveBtn)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(2, 10)
        self.verticalLayout_2.setStretch(4, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(2, 5)
        # 将所有内容放到MainWindow中
        MainWindow.setCentralWidget(self.centralwidget)
        # 上菜单
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1148, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSaveResult = QtWidgets.QAction(MainWindow)
        self.actionSaveResult.setObjectName("actionSaveResult")
        self.actioninterface = QtWidgets.QAction(MainWindow)
        self.actioninterface.setObjectName("actioninterface")
        self.actionDocumentation = QtWidgets.QAction(MainWindow)
        self.actionDocumentation.setObjectName("actionDocumentation")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menu.addAction(self.actionSaveResult)
        self.menu_2.addAction(self.actioninterface)
        self.menu_3.addAction(self.actionDocumentation)
        self.menu_3.addAction(self.actionAbout)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车辆检测跟踪系统"))
        self.VideoOpenBtn.setText(_translate("MainWindow", "打开视频"))
        self.CameraOpenBtn.setText(_translate("MainWindow", "打开摄像机"))
        self.radioButtonDefogOpen.setText(_translate("MainWindow", "去雾开关"))
        self.groupBox.setTitle(_translate("MainWindow", "权重设置"))
        self.YoloWeightsBtn.setText(_translate("MainWindow", "yolo权重"))
        self.StrongsortWeightsBtn.setText(_translate("MainWindow", "deepsort权重"))
        # self.ConfigStrongsortBtn.setText(_translate("MainWindow", "config_strongsort"))
        self.groupBox_2.setTitle(_translate("MainWindow", "测试结果目录设置"))
        self.labelproject.setText(_translate("MainWindow", "测试结果放置文件夹"))
        self.labelname.setText(_translate("MainWindow", "测试结果文件夹名称"))
        self.groupBox_3.setTitle(_translate("MainWindow", "参数设置"))
        self.labelimgsz.setText(_translate("MainWindow", "输入图片的大小"))
        self.labelconf_thres.setText(_translate("MainWindow", "置信度阈值"))
        self.labeliou_thres.setText(_translate("MainWindow", "nms的iou阈值"))
        self.labelmax_det.setText(_translate("MainWindow", "图片最多目标数量"))
        self.labelline_thickness.setText(_translate("MainWindow", "框线宽度（pixels）"))
        self.groupBox_4.setTitle(_translate("MainWindow", "功能开关"))
        self.checkBoxshow_vid.setText(_translate("MainWindow", "结果展示"))
        self.checkBoxsave_txt.setText(_translate("MainWindow", "坐标保存"))
        self.checkBoxsave_conf.setText(_translate("MainWindow", "置信度保存"))
        self.checkBoxsave_crop.setText(_translate("MainWindow", "目标保存"))
        self.checkBoxsave_vid.setText(_translate("MainWindow", "预测结果保存"))
        self.checkBoxnosave.setText(_translate("MainWindow", "不保存预测结果"))
        self.checkBoxagnostic_nms.setText(_translate("MainWindow", "agnostic_nms"))
        self.checkBoxaugment.setText(_translate("MainWindow", "augment"))
        self.checkBoxvisualize.setText(_translate("MainWindow", "visualize"))
        self.checkBoxupdate.setText(_translate("MainWindow", "update"))
        self.checkBoxexist_ok.setText(_translate("MainWindow", "exist_ok"))
        self.checkBoxhide_labels.setText(_translate("MainWindow", "hide_labels"))
        self.checkBoxhide_conf.setText(_translate("MainWindow", "hide_conf"))
        self.checkBoxhide_class.setText(_translate("MainWindow", "hide_class"))
        self.checkBoxhide_speed.setText(_translate("MainWindow", "hide_speed"))
        self.checkBoxhalf.setText(_translate("MainWindow", "half"))
        self.checkBoxdnn.setText(_translate("MainWindow", "dnn"))
        self.pushButtonParameterSet.setText(_translate("MainWindow", "更新参数"))
        # self.StartTrackBtn.setText(_translate("MainWindow", "开始跟踪"))
        self.labelshowresult.setText(_translate("MainWindow", "结果展示"))
        self.OutputVideoSaveBtn.setText(_translate("MainWindow", "开始记录\n"
                                                                 "结果"))
        self.OutputSaveBtn.setText(_translate("MainWindow", "导出当前\n"
                                                            "视频帧图片"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "设置"))
        self.menu_3.setTitle(_translate("MainWindow", "帮助"))
        self.actionSaveResult.setText(_translate("MainWindow", "SaveResult"))
        self.actioninterface.setText(_translate("MainWindow", "Interface"))
        self.actionDocumentation.setText(_translate("MainWindow", "Documentation"))
        self.actionAbout.setText(_translate("MainWindow", "About"))

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
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images

        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        # 使用wencam
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        # webcam = self.webcam

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
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
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
            self.CameraOpenBtn.setText("停止")
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
            if webcam:
                im = im.squeeze(0)

            # im = deHazeDefogging(im)

            # im0s = cv2.add(im0s,color_polygons_image)

            im = torch.from_numpy(im).to(device)
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

                annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
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
                    SpeedOver = False
                    if len(outputs[i]) > 0:
                        # j第几个框，output结果，conf置信度
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                            # 321.00000, 199.00000, 386.00000, 271.00000, 1.00000, 0.00000, 0.95377
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
                            bbox_speed, SpeedOverFlag = Estimated_speed(outputs_prev[-2], output, id, fps, bbox_width)
                            if SpeedOverFlag:
                                SpeedOver = True

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                bbox_conf = output[6]
                                result = ('%g ' * 11 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i, bbox_conf) \
                                    if save_conf \
                                    else ('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i )


                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(result)

                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                # label = None if hide_labels else (f'{id} {names[c]} ' if hide_conf else (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else (
                                #     f'{id} {conf:.2f}' if hide_class else f'{id} {bbox_speed}'))
                                label = f'{id}'
                                if not hide_conf:
                                    label += f'{conf:.2f} \t'
                                if not hide_class:
                                    label +=f'{names[c]} \t'
                                if not hide_speed:
                                    label += f'{bbox_speed}'
                                label = None if hide_labels else label
                                # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else (
                                #     f'{id} {conf:.2f}' if hide_class else (f'{id} {conf:.2f} {names[c]}' if hide_speed else f'{id} {names[c]} {conf:.2f} {bbox_speed}')))

                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                        c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                    # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')
                    self.statusBar.showMessage(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)', 500)
                    if SpeedOver:
                        self.Append(f"{s}Done.有疾驶车辆，请小心驾驶")

                else:
                    strongsort_list[i].increment_ages()
                    # LOGGER.info('No detections')
                    self.statusBar.showMessage('No detections', 500)

                # Stream results
                # 主要修改的地方
                self.im0 = annotator.result()
                if show_vid:
                    # 去雾开关，增加对比
                    if self.DefogOpen and webcam == False:
                        im0modify = cv2.cvtColor(im0s, cv2.COLOR_BGR2BGRA)

                        self.result = HistogramEqualization(self.im0)
                        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2BGRA)
                        resulttmp = np.zeros((im0modify.shape[0], im0modify.shape[1] * 2, 4))
                        resulttmp[:, :im0modify.shape[1], :] = im0modify.copy()
                        resulttmp[:, im0modify.shape[1]:, :] = self.result.copy()
                        resulttmp = np.array(resulttmp, dtype=np.uint8)
                        self.result = resulttmp
                    else:
                        self.result = cv2.cvtColor(self.im0, cv2.COLOR_BGR2BGRA)
                    # self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                              QtGui.QImage.Format_RGB32)
                    self.VideoShowLabel.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.VideoShowLabel.setScaledContents(True)
                    # cv2.imshow(str(p), self.im0)
                    # cv2.waitKey(1)  # 1 millisecond
                # 保存当前帧图片
                if self.save_image_flag:
                    # QtImgSave = QtGui.QPixmap.fromImage(self.QtImg)
                    # filename 文件目录 filetype 文件类型
                    filename, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "保存当前帧图片", './',
                                                                               "影像 (*.png *.jpg)")
                    if filename:
                        self.QtImg.save(filename)
                    self.save_image_flag = False

                if self.save_video_flag:
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
                # if self.cap.isOpened():
                #   self.cap.release()
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
