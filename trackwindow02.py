# -*- coding: utf-8 -*-
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


# 界面实现 PyQT GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import threading
from threading import Thread

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
class Ui_MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon("UI/icon.png"))



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 900)
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
        # ConfigStrongsor设置按钮
        self.ConfigStrongsortBtn = QtWidgets.QPushButton(self.groupBox)
        self.ConfigStrongsortBtn.setObjectName("ConfigStrongsortBtn")
        self.verticalLayout_6.addWidget(self.ConfigStrongsortBtn)
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
        # 置信度阈值 文本输入框
        self.conf_threslineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.conf_threslineEdit.setObjectName("conf_threslineEdit")
        self.verticalLayout_5.addWidget(self.conf_threslineEdit)
        # label nms的iou阈值
        self.labeliou_thres = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labeliou_thres.sizePolicy().hasHeightForWidth())
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
        # 分割线
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        # 开始跟踪按钮
        self.StartTrackBtn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartTrackBtn.sizePolicy().hasHeightForWidth())
        self.StartTrackBtn.setSizePolicy(sizePolicy)
        self.StartTrackBtn.setMinimumSize(QtCore.QSize(0, 0))
        self.StartTrackBtn.setObjectName("StartTrackBtn")
        self.verticalLayout.addWidget(self.StartTrackBtn)
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
        self.textEditShowResult = QtWidgets.QTextEdit(self.centralwidget)
        self.textEditShowResult.setObjectName("textEditShowResult")
        self.horizontalLayout_4.addWidget(self.textEditShowResult)
        # 分割线
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout_4.addWidget(self.line_5)
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
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.VideoOpenBtn.setText(_translate("MainWindow", "打开视频"))
        self.CameraOpenBtn.setText(_translate("MainWindow", "打开摄像机"))
        self.radioButtonDefogOpen.setText(_translate("MainWindow", "去雾开关"))
        self.groupBox.setTitle(_translate("MainWindow", "权重设置"))
        self.YoloWeightsBtn.setText(_translate("MainWindow", "yolo权重"))
        self.StrongsortWeightsBtn.setText(_translate("MainWindow", "deepsort权重"))
        self.ConfigStrongsortBtn.setText(_translate("MainWindow", "config_strongsort"))
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
        self.StartTrackBtn.setText(_translate("MainWindow", "开始跟踪"))
        self.labelshowresult.setText(_translate("MainWindow", "结果展示"))
        self.OutputSaveBtn.setText(_translate("MainWindow", "导出当前\n"
                                                            "视频帧图片"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "设置"))
        self.menu_3.setTitle(_translate("MainWindow", "帮助"))
        self.actionSaveResult.setText(_translate("MainWindow", "SaveResult"))
        self.actioninterface.setText(_translate("MainWindow", "Interface"))
        self.actionDocumentation.setText(_translate("MainWindow", "Documentation"))
        self.actionAbout.setText(_translate("MainWindow", "About"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
