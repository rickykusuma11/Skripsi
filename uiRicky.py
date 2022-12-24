from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QLabel, QFileDialog
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from tqdm import tqdm
import tensorflow as tf
import sys
import keyboard
import pickle
import cv2
import numpy as np
import sklearn

class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super(Dashboard, self).__init__()
        self.title = 'Home Aplikasi'
        uic.loadUi('UI/home1.ui', self)
        self.setWindowTitle(self.title)
        self.timer = QTimer()
        self.image = None
        self.processedImage = None
        self.Deteksi.clicked.connect(self.deteksi)
        self.Help.clicked.connect(self.Helpb)
        self.About.clicked.connect(self.Aboutb)
        self.Exit.clicked.connect(self.quitApplication)

    def Home(self):
        self.title = 'Home Aplikasi'
        uic.loadUi('UI/Home1.ui', self)
        self.setWindowTitle(self.title)
        self.timer = QTimer()
        self.Deteksi.clicked.connect(self.deteksi)
        self.Help.clicked.connect(self.Helpb)
        self.About.clicked.connect(self.Aboutb)
        self.Exit.clicked.connect(self.quitApplication)

    def deteksi(self):
        self.title = 'Deteksi'
        uic.loadUi('UI/Deteksi1.ui', self)
        self.setWindowTitle(self.title)
        self.timer = QTimer()
        self.image = None
        self.processedImage = None
        self.label1 = self.findChild(QLabel, "image1")
        self.label2 = self.findChild(QLabel, "image2")
        self.Select.clicked.connect(self.Selectfile)
        self.Prediksi.clicked.connect(self.detection)
        self.Save.clicked.connect(self.Savefile)
        self.Back.clicked.connect(self.Home)

    def Helpb(self):
        self.title = 'Help'
        uic.loadUi('UI/Help1.ui', self)
        self.setWindowTitle(self.title)
        self.Back1.clicked.connect(self.Home)

    def Aboutb(self):
        self.title = 'About'
        uic.loadUi('UI/About1.ui', self)
        self.setWindowTitle(self.title)
        self.Back2.clicked.connect(self.Home)

    def quitApplication(self):
        "Keluar aplikasi"
        userReply = QMessageBox.question(
            self, 'Exit', "Apakah Kamu Ingin Menutup Aplikasi Ini", QMessageBox.Yes | QMessageBox.No)
        if userReply == QMessageBox.Yes:
            keyboard.press_and_release('alt+F4')

    def Selectfile(self):
        fname, filter = QFileDialog.getOpenFileName(
            self, 'Open File', '.\Program\Test', "Image Files (*.jpg)")
        if fname:
            self.loadImage(fname)
        else:
            print('Batal')

    def detection(self, fname, confidence=0.9, iou_thresh=0.1):

        print("Memulai Prediksi. . .")
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(self.image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        sel_rects = rects[:500]