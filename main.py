import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import random
import time

from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, \
    QFileDialog, QLabel, QRubberBand, QComboBox, QMenu, QMainWindow, QAction, QProgressBar
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor, QPalette, QBrush, QIcon
from PyQt5.QtCore import QPoint, QRect, QSize, pyqtSignal, Qt, pyqtSlot
import sys
from utils import ImageContainer, xml_root, instance_to_xml, prediction, load_image, globWithTypes
from lxml import etree
from enum import Enum
from keras.models import load_model
import tensorflow as tf
from glob import glob
import os
import shutil
import threading
import cv2
import numpy as np
import math
from PIL import Image


class Mode(Enum):
    LABELING = 0
    CORRECTION = 1


class CorrectionMode(Enum):
    MOVE = 0
    RESIZE = 1
    OTHER = -1


class ResizeMode(Enum):
    TOPLEFT = 0
    TOP = 1
    TOPRIGHT = 2
    RIGHT = 3
    BOTTOMRIGHT = 4
    BOTTOM = 5
    BOTTOMLEFT = 6
    LEFT = 7
    OTHER = -1


class AppString(Enum):
    TITLE = 'Labeling tool'
    LOADFILE = 'File'
    LOADFOLDER = 'Folder'
    LOADVIDEO = 'Video'
    SAVE = 'Save'
    DELETE = 'Delete'
    AUTOLABEL = 'AutoLabel'
    EXIT = 'Exit'
    DESCRIPTION = 'Description'


class Label(Enum):
    SHIP = 'Ship'
    SPEEDBOAT = 'Speed boat'
    SAILBOAT = 'Sail boat'
    BUOY = 'Buoy'
    OTHER = 'Other'


class Utils:

    @staticmethod
    def changeCursor(cursorShape):
        QApplication.setOverrideCursor(QCursor(cursorShape))


class BoundingBox(QRubberBand):

    def __init__(self, shape, parent, label):
        super().__init__(shape, parent)
        self.pointCheckRange = 3
        self.canvasPositionRatio = (0, 0)
        self.canvasBoxRatio = (0, 0)
        self.label = label

    def pointOnTopLeft(self, pos):
        return (self.x() <= pos.x() < self.x() + self.pointCheckRange) and \
               (self.y() <= pos.y() < self.y() + self.pointCheckRange)

    def pointOnTop(self, pos):
        return (self.x() + self.pointCheckRange <= pos.x() < self.x() + self.width() - self.pointCheckRange) and \
               (self.y() <= pos.y() < self.y() + self.pointCheckRange)

    def pointOnTopRight(self, pos):
        return (self.x() + self.width() - self.pointCheckRange <= pos.x() < self.x() + self.width()) and \
               (self.y() <= pos.y() < self.y() + self.pointCheckRange)

    def pointOnRight(self, pos):
        return (self.x() + self.width() - self.pointCheckRange <= pos.x() < self.x() + self.width()) and \
               (self.y() + self.pointCheckRange <= pos.y() < self.y() + self.height() - self.pointCheckRange)

    def pointOnBottomRight(self, pos):
        return (self.x() + self.width() - self.pointCheckRange <= pos.x() < self.x() + self.width()) and \
               (self.y() + self.height() - self.pointCheckRange <= pos.y() < self.y() + self.height())

    def pointOnBottom(self, pos):
        return (self.x() + self.pointCheckRange <= pos.x() < self.x() + self.width() - self.pointCheckRange) and \
               (self.y() + self.height() - self.pointCheckRange <= pos.y() < self.y() + self.height())

    def pointOnBottomLeft(self, pos):
        return (self.x() <= pos.x() < self.x() + self.pointCheckRange) and \
               (self.y() + self.height() - self.pointCheckRange <= pos.y() < self.y() + self.height())

    def pointOnLeft(self, pos):
        return (self.x() <= pos.x() < self.x() + self.pointCheckRange) and \
               (self.y() + self.pointCheckRange <= pos.y() < self.y() + self.height() - self.pointCheckRange)


class Viewer(QLabel):
    changeBoxNum = pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent)

        self.setMouseTracking(True)
        self.__boxes = []
        self.selectedIdx = -1
        self.drawingThreshold = 30
        self.origin = QPoint()
        self.translateOffset = QPoint()
        self.__mode = Mode.LABELING
        self.__makeBoundingBox = False
        self.__correctionMode = CorrectionMode.OTHER
        self.resizeMode = ResizeMode.OTHER
        self.label = Label.SHIP
        self.colorTable = {Label.SHIP: Qt.blue,
                           Label.SPEEDBOAT: Qt.yellow,
                           Label.SAILBOAT: Qt.darkGray,
                           Label.BUOY: Qt.red,
                           Label.OTHER: Qt.green}
        self.__mouseLineVisible = True
        self.__InitializeMouseLine()
        self.__shiftFlag = False
        self.__resized = False

    def initialize(self):
        for box in self.__boxes:
            box.hide()
            box.deleteLater()

        self.__boxes.clear()
        self.selectedIdx = -1
        self.origin = QPoint()
        self.__mode = Mode.LABELING
        self.__makeBoundingBox = False
        self.__correctionMode = CorrectionMode.OTHER
        self.resizeMode = ResizeMode.OTHER
        self.label = Label.SHIP
        self.__resized = False

    def autoLabeling(self, boundingBoxes):
        for box in self.__boxes:
            box.hide()
            box.deleteLater()

        self.__boxes.clear()

        for idx, bbox in enumerate(boundingBoxes):
            x, y, w, h = bbox
            x = max(0, min(x, self.width()))
            y = max(0, min(y, self.height()))
            w = max(0, min(w, self.width()-x))
            h = max(0, min(h, self.width()-y))

            self.__boxes.append(BoundingBox(QRubberBand.Rectangle, self, Label.SHIP))
            self.__boxes[idx].setGeometry(QRect(x, y, w, h))
            self.__boxes[idx].geometry()
            self.__boxes[idx].setPalette(self.__boundingBoxColor(Label.SHIP)) # TODO - Multi classification Labeling
            self.__boxes[idx].show()

            self.__boxes[idx].canvasPositionRatio = \
                (self.__boxes[idx].pos().x() / self.width(), self.__boxes[idx].pos().y() / self.height())
            self.__boxes[idx].canvasBoxRatio = \
                (self.__boxes[idx].width() / self.width(), self.__boxes[idx].height() / self.height())

        self.changeBoxNum.emit(len(self.__boxes))

    @property
    def shiftFlag(self):
        return self.__shiftFlag

    @shiftFlag.setter
    def shiftFlag(self, newFlag):
        self.__shiftFlag = newFlag

    @property
    def mouseLineVisible(self):
        return self.__mouseLineVisible

    @mouseLineVisible.setter
    def mouseLineVisible(self, flag):
        self.__mouseLineVisible = flag

        if self.__mouseLineVisible:
            for mouseLine in self.__mouseLines:
                mouseLine.show()
        else:
            for mouseLine in self.__mouseLines:
                mouseLine.hide()

    @property
    def boxes(self):
        bndBox = []

        for box in self.__boxes:
            bndBox.append([box.x(), box.y(), box.width(), box.height(), box.label])
        return bndBox

    @property
    def makeBoundingBox(self):
        return self.__makeBoundingBox

    @property
    def correctionMode(self):
        return self.__correctionMode

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, newMode):
        self.__mode = newMode

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            if self.__mode == Mode.LABELING:
                self.origin = QMouseEvent.pos()
                box = BoundingBox(QRubberBand.Line, self, self.label)
                box.setGeometry(QRect(self.origin, QSize()))
                box.geometry()
                box.setPalette(self.__boundingBoxColor())
                box.show()
                QHBoxLayout().addWidget(box)

                self.__boxes.insert(0, box)
                self.__makeBoundingBox = True
            elif self.__mode == Mode.CORRECTION:
                selectedIdx, resizeMode = self.__findResizingBox(QMouseEvent)

                if selectedIdx >= 0:
                    self.__correctionMode = CorrectionMode.RESIZE
                    self.resizeMode = resizeMode
                else:
                    selectedIdx = self.__findCorrectionBox(QMouseEvent)

                    if selectedIdx >= 0:
                        self.__correctionMode = CorrectionMode.MOVE
                        Utils.changeCursor(Qt.ClosedHandCursor)
                        self.translateOffset = QMouseEvent.pos() - self.__boxes[selectedIdx].pos()
                self.selectedIdx = selectedIdx
        super().mousePressEvent(QMouseEvent)

    def mouseMoveEvent(self, QMouseEvent):
        self.__setMouseLinePosition(QMouseEvent.pos())

        if self.__mode == Mode.CORRECTION and self.__correctionMode == CorrectionMode.OTHER:
            self.__findResizingBox(QMouseEvent)

        if self.__mode == Mode.LABELING and self.__resized:
            if self.rect().contains(QMouseEvent.pos()):
                self.mouseLineVisible = True
                self.__setMouseLinePosition(QMouseEvent.pos())
                self.__resized = False

        if self.__makeBoundingBox:
            clipCoord = self.__clipCoordinateInWidget(QMouseEvent)
            self.__boxes[0].setGeometry(QRect(self.origin, clipCoord).normalized())
            self.__boxes[0].geometry()
        elif self.__correctionMode != CorrectionMode.OTHER:
            selectedBox = self.__boxes[self.selectedIdx]
            if self.__correctionMode == CorrectionMode.RESIZE:

                newX, newY, newW, newH = self.__getResizeDimension(selectedBox, QMouseEvent.pos(), self.resizeMode)

                selectedBox.setGeometry(QRect(newX, newY, newW, newH))
                selectedBox.geometry()

            elif self.__correctionMode == CorrectionMode.MOVE:
                nextCenterPosition = QMouseEvent.pos() - self.translateOffset
                nextCenterPosition.setX(max(0, min(nextCenterPosition.x(), self.width() - selectedBox.width())))
                nextCenterPosition.setY(max(0, min(nextCenterPosition.y(), self.height() - selectedBox.height())))
                selectedBox.move(nextCenterPosition)
        super().mouseMoveEvent(QMouseEvent)

    def mouseReleaseEvent(self, QMouseEvent):
        Utils.changeCursor(Qt.ArrowCursor)
        if self.__makeBoundingBox:
            if self.__boxes[0].width() * self.__boxes[0].height() < self.drawingThreshold:
                self.removeBoundingBox(0)
            else:
                self.__boxes[0].canvasPositionRatio = (self.__boxes[0].pos().x() / self.width(), self.__boxes[0].pos().y() / self.height())
                self.__boxes[0].canvasBoxRatio = (self.__boxes[0].width() / self.width(), self.__boxes[0].height() / self.height())
                self.changeBoxNum.emit(len(self.__boxes))
            self.__makeBoundingBox = False

        if self.__correctionMode != CorrectionMode.OTHER:
            selectedBox = self.__boxes[self.selectedIdx]
            if self.__correctionMode == CorrectionMode.RESIZE:
                selectedBox.canvasBoxRatio = (selectedBox.width() / self.width(), selectedBox.height() / self.height())
                selectedBox.canvasPositionRatio = (selectedBox.pos().x() / self.width(), selectedBox.pos().y() / self.height())

                if selectedBox.width() * selectedBox.height() < self.drawingThreshold:
                    self.removeBoundingBox(self.selectedIdx)

                self.resizeMode = ResizeMode.OTHER

            elif self.__correctionMode == CorrectionMode.MOVE:
                selectedBox.canvasPositionRatio = (selectedBox.pos().x() / self.width(), selectedBox.pos().y() / self.height())

            # TODO - Remove Invalid Bounding boxes with Area Threshold or Some Rules
            self.__correctionMode = CorrectionMode.OTHER

        if self.__shiftFlag:
            self.__shiftFlag = False
            self.mode = Mode.LABELING
            self.mouseLineVisible = True

        super().mouseReleaseEvent(QMouseEvent)

    def resizeEvent(self, QResizeEvent):
        if QResizeEvent.oldSize().isValid():
            newSize = QResizeEvent.size()
            for box in self.__boxes:
                box.resize(newSize.width() * box.canvasBoxRatio[0], newSize.height() * box.canvasBoxRatio[1])
                box.move(newSize.width() * box.canvasPositionRatio[0], newSize.height() * box.canvasPositionRatio[1])

            self.__resized = True
        super().resizeEvent(QResizeEvent)

    def leaveEvent(self, QEvent):
        if self.mode == Mode.LABELING:
            self.mouseLineVisible = False

    def enterEvent(self, QEvent):
        if self.mode == Mode.LABELING:
            self.mouseLineVisible = True
            self.__setMouseLinePosition(QEvent.pos())

    def contextMenuEvent(self, event):
        selectedIdx = self.__findCorrectionBox(event.pos())

        if selectedIdx >= 0:
            contextMenu = QMenu(self)

            for label in Label:
                pixmap = QPixmap(12, 12)
                pixmap.fill(QColor(self.colorTable[label]))
                contextMenu.addAction(QIcon(pixmap), label.value)

            contextMenu.addAction(AppString.DELETE.value)

            action = contextMenu.exec_(self.mapToGlobal(event.pos()))

            if action is not None:
                if action.text() == AppString.DELETE.value:
                    self.removeBoundingBox(selectedIdx)
                else:
                    selectedBox = self.__boxes[selectedIdx]
                    selectedBox.label = Label(action.text())
                    selectedBox.setPalette(self.__boundingBoxColor(Label(action.text())))

    def setLabel(self, newLabel):
        self.label = Label(newLabel)

    def removeBoundingBox(self, idx=None):
        if idx is None:
            idx = self.selectedIdx
            self.selectedIdx = -1

        if 0 <= idx < len(self.__boxes):
            self.__boxes[idx].hide()
            self.__boxes[idx].deleteLater()
            self.__boxes.pop(idx)
            self.changeBoxNum.emit(len(self.__boxes))

    def __setMouseLinePosition(self, position):
        self.__mouseLines[0].setGeometry(QRect(QPoint(position.x(), 0), QPoint(position.x(), position.y())))
        self.__mouseLines[1].setGeometry(
            QRect(QPoint(position.x(), position.y()), QPoint(position.x(), self.height())))
        self.__mouseLines[2].setGeometry(QRect(QPoint(0, position.y()), QPoint(position.x(), position.y())))
        self.__mouseLines[3].setGeometry(
            QRect(QPoint(position.x(), position.y()), QPoint(self.width(), position.y())))

    def __InitializeMouseLine(self):
        self.__mouseLines = [QRubberBand(QRubberBand.Line, self) for _ in range(4)]

        for mouseLine in self.__mouseLines:
            mouseLine.setGeometry(QRect(QPoint(0, 0), QPoint(0, 0)))
            mouseLine.setPalette(self.__boundingBoxColor(Label.OTHER))
            mouseLine.show()

    def __boundingBoxColor(self, label=None):
        if label is None:
            label = self.label

        color = QColor(self.colorTable[label])
        palette = QPalette()
        color.setAlpha(80)
        palette.setBrush(QPalette.Highlight, QBrush(color))

        return palette

    def __getResizeDimension(self, box, mousePos, resizeMode):
        oldTopLeftX, oldTopLeftY = box.pos().x(), box.pos().y()
        oldBottomRightX, oldBottomRightY = oldTopLeftX + box.width(), oldTopLeftY + box.height()
        oldWidth, oldHeight = box.width(), box.height()

        mousePos = self.__clipCoordinateInWidget(mousePos)

        if resizeMode == ResizeMode.TOPLEFT:
            newX, newY = mousePos.x(), mousePos.y()
            newW, newH = oldBottomRightX - mousePos.x(), oldBottomRightY - mousePos.y()

            if newW < 0 and newH < 0:
                self.resizeMode = ResizeMode.BOTTOMRIGHT
                newH = 0
                newW = 0
                newY = oldBottomRightY
                newX = oldBottomRightX
            elif newW < 0:
                self.resizeMode = ResizeMode.TOPRIGHT
                newW = 0
                newX = oldBottomRightX
            elif newH < 0:
                self.resizeMode = ResizeMode.BOTTOMLEFT
                newH = 0
                newY = oldBottomRightY
        elif resizeMode == ResizeMode.TOP:
            newX, newY = oldTopLeftX, mousePos.y()
            newW, newH = oldWidth, oldBottomRightY - mousePos.y()

            if newH < 0:
                self.resizeMode = ResizeMode.BOTTOM
                newH = 0
                newY = oldBottomRightY
        elif resizeMode == ResizeMode.TOPRIGHT:
            newX, newY = oldTopLeftX, mousePos.y()
            newW, newH = mousePos.x() - oldTopLeftX, oldBottomRightY - mousePos.y()

            if newW < 0 and newH < 0:
                self.resizeMode = ResizeMode.BOTTOMLEFT
                newW = 0
                newH = 0
                newX = oldTopLeftX
                newY = oldBottomRightY
            elif newW < 0:
                self.resizeMode = ResizeMode.TOPLEFT
                newW = 0
                newX = oldTopLeftX
            elif newH < 0:
                self.resizeMode = ResizeMode.BOTTOMRIGHT
                newH = 0
                newY = oldBottomRightY
        elif resizeMode == ResizeMode.RIGHT:
            newX, newY = oldTopLeftX, oldTopLeftY
            newW, newH = mousePos.x() - oldTopLeftX, oldHeight

            if newW < 0:
                self.resizeMode = ResizeMode.LEFT
                newW = 0
                newX = oldTopLeftX
        elif resizeMode == ResizeMode.BOTTOMRIGHT:
            newX, newY = oldTopLeftX, oldTopLeftY
            newW, newH = mousePos.x() - oldTopLeftX, mousePos.y() - oldTopLeftY

            if newW < 0 and newH < 0:
                self.resizeMode = ResizeMode.TOPLEFT
                newW = 0
                newH = 0
                newY = oldTopLeftY
                newX = oldTopLeftX
            elif newW < 0:
                self.resizeMode = ResizeMode.BOTTOMLEFT
                newW = 0
                newX = oldTopLeftX
            elif newH < 0:
                self.resizeMode = ResizeMode.TOPRIGHT
                newH = 0
                newY = oldTopLeftY
        elif resizeMode == ResizeMode.BOTTOM:
            newX, newY = oldTopLeftX, oldTopLeftY
            newW, newH = oldWidth, mousePos.y() - oldTopLeftY

            if newH < 0:
                self.resizeMode = ResizeMode.TOP
                newH = 0
                newY = oldTopLeftY
        elif resizeMode == ResizeMode.BOTTOMLEFT:
            newX, newY = mousePos.x(), oldTopLeftY
            newW, newH = oldBottomRightX - mousePos.x(), mousePos.y() - oldTopLeftY

            if newW < 0 and newH < 0:
                self.resizeMode = ResizeMode.TOPRIGHT
                newW = 0
                newH = 0
                newX = oldBottomRightX
                newY = oldTopLeftY
            elif newW < 0:
                self.resizeMode = ResizeMode.BOTTOMRIGHT
                newW = 0
                newX = oldBottomRightX
            elif newH < 0:
                self.resizeMode = ResizeMode.TOPLEFT
                newH = 0
                newY = oldTopLeftY
        elif resizeMode == ResizeMode.LEFT:
            newX, newY = mousePos.x(), oldTopLeftY
            newW, newH = oldBottomRightX - mousePos.x(), oldHeight

            if newW < 0:
                self.resizeMode = ResizeMode.RIGHT
                newW = 0
                newX = oldBottomRightX

        return (newX, newY, newW, newH)

    def __findResizingBox(self, QMouseEvent):
        for idx, box in enumerate(self.__boxes):
            resizeMode = self.__mouseOnEdge(box, QMouseEvent)
            if resizeMode != ResizeMode.OTHER:
                return idx, resizeMode
        return -1, ResizeMode.OTHER

    def __mouseOnEdge(self, box:BoundingBox, QMouseEvent):
        if box.pointOnTopLeft(QMouseEvent):
            Utils.changeCursor(Qt.SizeFDiagCursor)
            return ResizeMode.TOPLEFT
        elif box.pointOnTop(QMouseEvent):
            Utils.changeCursor(Qt.SizeVerCursor)
            return ResizeMode.TOP
        elif box.pointOnTopRight(QMouseEvent):
            Utils.changeCursor(Qt.SizeBDiagCursor)
            return ResizeMode.TOPRIGHT
        elif box.pointOnRight(QMouseEvent):
            Utils.changeCursor(Qt.SizeHorCursor)
            return ResizeMode.RIGHT
        elif box.pointOnBottomRight(QMouseEvent):
            Utils.changeCursor(Qt.SizeFDiagCursor)
            return ResizeMode.BOTTOMRIGHT
        elif box.pointOnBottom(QMouseEvent):
            Utils.changeCursor(Qt.SizeVerCursor)
            return ResizeMode.BOTTOM
        elif box.pointOnBottomLeft(QMouseEvent):
            Utils.changeCursor(Qt.SizeBDiagCursor)
            return ResizeMode.BOTTOMLEFT
        elif box.pointOnLeft(QMouseEvent):
            Utils.changeCursor(Qt.SizeHorCursor)
            return ResizeMode.LEFT
        else:
            Utils.changeCursor(Qt.ArrowCursor)
            return ResizeMode.OTHER

    def __clipCoordinateInWidget(self, QMouseEvent):
        clipCoord = QPoint()
        clipCoord.setX(max(0, min(QMouseEvent.x(), self.width())))
        clipCoord.setY(max(0, min(QMouseEvent.y(), self.height())))

        return clipCoord

    def __findCorrectionBox(self, QMouseEvent):
        for idx, box in enumerate(self.__boxes):
            if self.__mouseInBox(QMouseEvent, box):
                return idx
        return -1

    def __mouseInBox(self, QMouseEvent, box):
        inX = box.x() <= QMouseEvent.x() < box.x() + box.width()
        inY = box.y() <= QMouseEvent.y() < box.y() + box.height()
        return inX and inY


class MainUI(object):

    def __init__(self):
        self.windowWidth = 1000
        self.windowHeight = 800
        self.windowTitle = AppString.TITLE.value
        self.windowXPos = 300
        self.windowYPos = 200

        self.allowImageType = '(*.jpg *.png *.jpeg)'
        self.allowVideoType = '(*.mp4 *.avi)'

    def setupUi(self):
        self.loadFileBtn = QAction(QIcon('./icon/file-add-outline.svg'), AppString.LOADFILE.value, self)
        self.loadFileBtn.setIconText(AppString.LOADFILE.value)
        self.loadFolderBtn = QAction(QIcon('./icon/folder-add-outline.svg'), AppString.LOADFOLDER.value, self)
        self.loadFolderBtn.setIconText(AppString.LOADFOLDER.value)
        self.loadVideoBtn = QAction(QIcon('./icon/film-outline.svg'), AppString.LOADVIDEO.value, self)
        self.loadVideoBtn.setIconText(AppString.LOADVIDEO.value)
        self.saveBtn = QAction(QIcon('./icon/download-outline.svg'), AppString.SAVE.value, self, shortcut="Ctrl+S")
        self.saveBtn.setIconText(AppString.SAVE.value)
        self.autoLabelBtn = QAction(QIcon('./icon/crop-outline.svg'), AppString.AUTOLABEL.value, self)
        self.autoLabelBtn.setIconText(AppString.AUTOLABEL.value)
        self.description = QAction(QIcon('./icon/question-mark-outline.svg'), AppString.DESCRIPTION.value, self)
        self.description.setIconText(AppString.DESCRIPTION.value)
        self.labelComboBox = QComboBox()

        self.remaining = QLabel('| Remaining: ')
        self.pbarLoad = QProgressBar(self)
        self.pbarLoad.setFixedWidth(300)
        self.imageIdx = QLabel(' 1/120')

        remainingLayout = QHBoxLayout()
        remainingLayout.addWidget(self.remaining, 0)
        remainingLayout.addWidget(self.pbarLoad, 0)
        remainingLayout.addWidget(self.imageIdx, 1)
        remainingLayout.setContentsMargins(0, 0, 0, 0)
        self.remainingNotification = QWidget()
        self.remainingNotification.setLayout(remainingLayout)
        self.remainingNotification.hide()

        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.setMovable(False)
        self.toolbar.addActions([self.loadFileBtn, self.loadFolderBtn, self.loadVideoBtn, self.saveBtn, self.autoLabelBtn, self.description])

        for action in self.toolbar.actions():
            widget = self.toolbar.widgetForAction(action)
            widget.setFixedSize(85, 60)

        self.toolbar.addWidget(self.labelComboBox)

        self.toolbar.setIconSize(QSize(30, 30))
        self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon | Qt.AlignLeading)
        self.toolbar.setStyleSheet('QToolBar {padding-right: 30px;}')

        self.viewer = Viewer(self)
        self.viewer.setScaledContents(True)
        self.viewer.setFocusPolicy(Qt.StrongFocus)

        self.notification = QLabel(Mode.LABELING.name, self)
        self.notification.setStyleSheet('background-color: rgb(0, 255, 0)')
        self.boundingBoxNum = QLabel('| Box: 0')

        self.description = QLabel('')
        self.pbar = QProgressBar(self)
        self.pbar.setMaximumWidth(300)
        self.pbar.hide()

        self.bottomBar = self.statusBar()
        self.bottomBar.setStyleSheet("background-color: rgb(200, 200, 200)")
        self.bottomBar.addWidget(self.notification)
        self.bottomBar.addWidget(self.boundingBoxNum)
        self.bottomBar.addWidget(self.remainingNotification)
        self.bottomBar.addPermanentWidget(self.description)
        self.bottomBar.addPermanentWidget(self.pbar)

        self.setCentralWidget(self.viewer)
        self.setGeometry(self.windowXPos, self.windowYPos, self.windowWidth, self.windowHeight)
        self.setWindowTitle(self.windowTitle)


class Labeling(QMainWindow, MainUI):
    def __init__(self):
        super().__init__()
        Utils.changeCursor(Qt.WaitCursor)
        self.setupUi()
        self.setWindowIcon(QIcon('./icon/favicon.png'))
        self.setMinimumSize(self.windowWidth, self.windowHeight)
        self.loadFileBtn.triggered.connect(self.openFileDialogue)
        self.loadFolderBtn.triggered.connect(self.openFolderDialogue)
        self.loadVideoBtn.triggered.connect(self.openVideoDiaglogue)
        self.saveBtn.triggered.connect(self.saveFileDialogue)
        self.autoLabelBtn.triggered.connect(self.autoLabel)
        self.viewer.changeBoxNum.connect(self.changeBoxNum)

        for label in Label:
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(self.viewer.colorTable[label]))
            self.labelComboBox.addItem(QIcon(pixmap), label.value)

        self.labelComboBox.setCurrentIndex(len(Label)-1)
        self.labelComboBox.currentTextChanged.connect(self.viewer.setLabel)
        self.labelComboBox.setCurrentIndex(0)
        self.show()
        self.setFocus()
        self.loadImage = None
        self.getMultipleInput = False
        self.yolo = load_model('./yolov2_ship_model.h5', custom_objects={'tf': tf})
        Utils.changeCursor(Qt.ArrowCursor)

    def initialize(self):
        self.labelComboBox.setCurrentIndex(0)
        self.changeBoxNum(0)
        self.pbarLoad.setValue(0)
        self.loadImage = None
        self.remainingNotification.hide()
        self.imageIdx.setText('')

    def keyPressEvent(self, QKeyEvent):
        if not self.viewer.makeBoundingBox and self.viewer.correctionMode == CorrectionMode.OTHER:
            if QKeyEvent.key() == Qt.Key_I:
                self.viewer.mode = Mode.CORRECTION
                self.viewer.mouseLineVisible = False
            elif QKeyEvent.key() == Qt.Key_Escape:
                self.viewer.mode = Mode.LABELING
                self.viewer.mouseLineVisible = True
            elif QKeyEvent.key() == Qt.Key_Shift:
                self.viewer.mode = Mode.CORRECTION
                self.viewer.mouseLineVisible = False
            self.__changeModeLabel(self.viewer.mode)

        if QKeyEvent.key() == Qt.Key_Delete:
            if self.viewer.mode == Mode.CORRECTION:
                self.viewer.removeBoundingBox()

        super().keyPressEvent(QKeyEvent)

    def keyReleaseEvent(self, QKeyEvent):
        if not self.viewer.makeBoundingBox and self.viewer.correctionMode == CorrectionMode.OTHER:
            if QKeyEvent.key() == Qt.Key_Shift:
                self.viewer.mode = Mode.LABELING
                self.viewer.mouseLineVisible = True
                self.__changeModeLabel(self.viewer.mode)
                Utils.changeCursor(Qt.ArrowCursor)
        elif self.viewer.correctionMode != CorrectionMode.OTHER:
            if QKeyEvent.key() == Qt.Key_Shift:
                self.__changeModeLabel(Mode.LABELING)
                self.viewer.shiftFlag = True

    @pyqtSlot(int)
    def changeBoxNum(self, num):
        self.boundingBoxNum.setText('| Box: {}'.format(num))

    def openVideoDiaglogue(self):
        videoPath, fileType = QFileDialog.getOpenFileName(self, 'Select Video', '',
                                                         'Video files {}'.format(self.allowVideoType),
                                                         options=QFileDialog.DontUseNativeDialog)

        if videoPath != '':
            self.initialize()
            self.viewer.initialize()
            Utils.changeCursor(Qt.WaitCursor)
            videoDir = self.__frame_extraction(videoPath)
            self.__multiInputLoading(videoDir)
            Utils.changeCursor(Qt.ArrowCursor)

    def openFileDialogue(self):
        imagePath, fileType = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files {}'.format(self.allowImageType), options=QFileDialog.DontUseNativeDialog)

        if imagePath != '':
            self.getMultipleInput = False
            rawImage = QImage(imagePath)
            self.initialize()
            self.viewer.initialize()
            self.loadImage = ImageContainer(rawImage, imagePath)
            self.viewer.setPixmap(QPixmap.fromImage(rawImage.scaled(self.viewer.width(), self.viewer.height())))

    def openFolderDialogue(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Directory', options=QFileDialog.DontUseNativeDialog)
        if directory != '':
            self.initialize()
            self.viewer.initialize()
            Utils.changeCursor(Qt.WaitCursor)
            self.__multiInputLoading(directory)
            Utils.changeCursor(Qt.ArrowCursor)

    def saveFileDialogue(self):
        if self.getMultipleInput:
            if self.loadImage is not None:
                xmlName = self.loadImage.fileName.split('.')[0] + '.xml'
                imageFullPath = os.path.join(self.imageSaveFolder, self.loadImage.fileName)
                annotationFullPath = os.path.join(self.annotationSaveFolder, xmlName)

                self.__saveToXml(annotationFullPath)
                shutil.move(self.loadImage.filePath, imageFullPath)

                threading.Thread(target=self.__threadMessage, args=('{} saved!'.format(xmlName),), name='Thread-SavedMessage').start()

                self.currentIdx += 1
                if self.currentIdx < len(self.imagePaths):
                    self.labelComboBox.setCurrentIndex(0)
                    self.changeBoxNum(0)
                    self.viewer.initialize()
                    rawImage = QImage(self.imagePaths[self.currentIdx])
                    self.loadImage = ImageContainer(rawImage, self.imagePaths[self.currentIdx])
                    self.viewer.setPixmap(QPixmap.fromImage(self.loadImage.image.scaled(self.viewer.width(), self.viewer.height())))
                    self.imageIdx.setText('{}/{}'.format(self.currentIdx+1, len(self.imagePaths)))
                    self.pbarLoad.setValue((self.currentIdx+1) * (100 / len(self.imagePaths)))
                else:
                    self.getMultipleInput = False
                    self.currentIdx = 0
                    self.loadImage = None

                    pixmap = QPixmap(self.viewer.width(), self.viewer.height())
                    pixmap.fill(QColor(Qt.gray))
                    self.viewer.setPixmap(pixmap)

                    self.initialize()
                    self.viewer.initialize()

        elif not self.getMultipleInput:
            if self.loadImage is not None:
                xmlName = self.loadImage.fileName.split('.')[0] + '.xml'
                savePath, fileType = QFileDialog.getSaveFileName(self, 'Save', xmlName, 'xml files {}'.format('*.xml'))

                if fileType != '':
                    if savePath.split('/')[-1].split('.')[-1] != 'xml':
                        savePath += '.xml'

                    self.__saveToXml(savePath)

                    pixmap = QPixmap(self.viewer.width(), self.viewer.height())
                    pixmap.fill(QColor(Qt.gray))
                    self.viewer.setPixmap(pixmap)

                    self.initialize()
                    self.viewer.initialize()

    def autoLabel(self):
        if self.loadImage is not None:
            Utils.changeCursor(Qt.WaitCursor)
            image = load_image(self.loadImage.filePath)
            boundingBoxes = prediction(image, self.yolo)

            for box in boundingBoxes:
                oldH, oldW, _ = image.shape
                newH, newW = self.viewer.height(), self.viewer.width()
                box[:] = [box[0]*newW/oldW, box[1]*newH/oldH, box[2]*newW/oldW, box[3]*newH/oldH]

            self.viewer.autoLabeling(boundingBoxes)
            Utils.changeCursor(Qt.ArrowCursor)

    def __multiInputLoading(self, dir):
        self.imageSaveFolder = os.path.join(dir, 'image')
        self.annotationSaveFolder = os.path.join(dir, 'annotation')
        os.makedirs(self.imageSaveFolder, exist_ok=True)
        os.makedirs(self.annotationSaveFolder, exist_ok=True)

        self.getMultipleInput = True
        self.imagePaths = globWithTypes(dir, ['png', 'jpg', 'jpeg'])

        if len(self.imagePaths) < 1:
            threading.Thread(target=self.__threadMessage, args=('Images not exist in {}'.format(dir),),
                             name='Thread-NoImageExist').start()
            return

        self.initialize()
        self.viewer.initialize()

        self.currentIdx = 0
        self.loadImage = ImageContainer(QImage(self.imagePaths[self.currentIdx]),
                                        self.imagePaths[self.currentIdx])
        self.viewer.setPixmap(QPixmap.fromImage(self.loadImage.image.scaled(self.viewer.width(), self.viewer.height())))

        self.remainingNotification.show()
        self.imageIdx.setText('{}/{}'.format(1, len(self.imagePaths)))
        self.pbarLoad.setValue(1 * (100 / len(self.imagePaths)))

    def __threadMessage(self, message):
        self.description.setText(message)
        time.sleep(2)
        self.description.setText('')

    def __saveToXml(self, filePath):
        bndBox = self.viewer.boxes
        annotation = xml_root(self.loadImage.fileName, self.loadImage.imageHeight, self.loadImage.imageWidth)

        instances = []

        for box in bndBox:
            xmin, ymin, width, height, label = box
            positionRatio = (xmin/self.viewer.width(), ymin/self.viewer.height())
            scaleRatio = (width/self.viewer.width(), height/self.viewer.height())

            bboxXmin = self.loadImage.imageWidth * positionRatio[0]
            bboxYmin = self.loadImage.imageHeight * positionRatio[1]
            bboxXmax = bboxXmin + self.loadImage.imageWidth * scaleRatio[0]
            bboxYmax = bboxYmin + self.loadImage.imageHeight * scaleRatio[1]

            bboxXmin = max(0, min(math.floor(bboxXmin), self.loadImage.imageWidth-1))
            bboxYmin = max(0, min(math.floor(bboxYmin), self.loadImage.imageHeight-1))
            bboxXmax = max(0, min(math.ceil(bboxXmax), self.loadImage.imageWidth-1))
            bboxYmax = max(0, min(math.ceil(bboxYmax), self.loadImage.imageHeight-1))

            instances.append({'bbox': [bboxXmin, bboxYmin, bboxXmax, bboxYmax],
                              'category_id': label.value})

        for instance in instances:
            annotation.append(instance_to_xml(instance))
        etree.ElementTree(annotation).write(filePath)

    def __changeModeLabel(self, mode):
        if mode == Mode.CORRECTION:
            self.notification.setText(Mode.CORRECTION.name)
            self.notification.setStyleSheet('QWidget { background-color: %s }' % (QColor(255, 0, 0).name()))
        elif mode == Mode.LABELING:
            self.notification.setText(Mode.LABELING.name)
            self.notification.setStyleSheet('QWidget { background-color: %s }' % (QColor(0, 255, 0).name()))

    def __frame_extraction(self, video_path):
        video_name = os.path.basename(video_path).split('.')[0]
        video_directory = os.path.dirname(video_path)
        destination_path = os.path.join(video_directory, video_name)
        os.makedirs(destination_path, exist_ok=True)

        self.description.setText('Frame extraction ')
        self.pbar.show()
        self.pbar.setValue(0)

        cnt = 0
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data_length = int(0.05 * length)
        using_idx = np.array(random.sample(range(length), data_length))

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                if cnt in using_idx:
                    Image.fromarray(frame[..., ::-1]).save(os.path.join(destination_path, '{}_{}.jpg'.format(video_name, cnt)))
                cnt += 1
                percent = (cnt + 1) / length * 100
                self.pbar.setValue(percent)
            else:
                break

        cap.release()

        self.description.setText('')
        self.pbar.hide()

        return destination_path


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Labeling()
    sys.exit(app.exec_())
