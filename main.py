from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QPushButton, \
    QFileDialog, QLabel, QRubberBand, QComboBox, QMenu, QShortcut, QMainWindow, QAction, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor, QPalette, QBrush, QPainter, QPen, QIcon, QKeySequence
from PyQt5.QtCore import QPoint, QRect, QSize, pyqtSignal, Qt, QObject, pyqtSlot
import sys
from utils import ImageContainer, xml_root, instance_to_xml
from lxml import etree
from enum import Enum


class Mode(Enum):
    LABELING = 0
    CORRECTION = 1
    OTHER = -1


class CorrectionMode(Enum):
    MOVE = 0
    RESIZE = 1
    OTHER = -1


class ResizeMode(Enum):
    TOPLEFT = 0
    TOP = 1
    TOPRIGHT = 2
    RIGHT = 3
    RIGHTBOTTOM = 4
    BOTTOM = 5
    BOTTOMLEFT = 6
    LEFT = 7
    OTHER = -1


class AppString(Enum):
    TITLE = 'Labeling tool'
    LOAD = 'Load'
    SAVE = 'Save'
    DELETE = 'Delete'
    AUTOLABEL = 'AutoLabel'
    EXIT = 'Exit'
    DESCRIPTION = 'Description'


class Label(Enum):
    SHIP = 'Ship'
    BUOY = 'Buoy'
    OTHER = 'Other'


class BoundingBox(QRubberBand):

    def __init__(self, QRubberBand_Shape, parent, label):
        super().__init__(QRubberBand_Shape, parent)
        self.pointCheckRange = 10
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

    def pointOnRightBottom(self, pos):
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
        self.colorTable = {Label.SHIP: Qt.blue, Label.BUOY: Qt.red, Label.OTHER: Qt.green}

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
                box = BoundingBox(QRubberBand.Rectangle, self, self.label)
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
                        self.__changeCursor(Qt.ClosedHandCursor)
                        self.translateOffset = QMouseEvent.pos() - self.__boxes[selectedIdx].pos()
                self.selectedIdx = selectedIdx

        super().mousePressEvent(QMouseEvent)

    def mouseMoveEvent(self, QMouseEvent):
        if self.__mode == Mode.CORRECTION and self.__correctionMode == CorrectionMode.OTHER:
            self.__findResizingBox(QMouseEvent)

        if self.__makeBoundingBox:
            clipCoord = self.__clipCoordinateInWidget(QMouseEvent)
            self.__boxes[0].setGeometry(QRect(self.origin, clipCoord).normalized())
            self.__boxes[0].geometry()
        elif self.__correctionMode != CorrectionMode.OTHER:
            selectedBox = self.__boxes[self.selectedIdx]
            if self.__correctionMode == CorrectionMode.RESIZE:

                newX, newY, newW, newH = self.__getResizeDimension(selectedBox, QMouseEvent.pos(), self.resizeMode)

                #TODO - MAKE REVERSIBLE
                selectedBox.setGeometry(QRect(newX, newY, newW, newH))
                selectedBox.geometry()

            elif self.__correctionMode == CorrectionMode.MOVE:
                nextCenterPosition = QMouseEvent.pos() - self.translateOffset
                nextCenterPosition.setX(max(0, min(nextCenterPosition.x(), self.width() - selectedBox.width() - 1)))
                nextCenterPosition.setY(max(0, min(nextCenterPosition.y(), self.height() - selectedBox.height() - 1)))
                selectedBox.move(nextCenterPosition)
        super().mouseMoveEvent(QMouseEvent)

    def mouseReleaseEvent(self, QMouseEvent):
        self.__changeCursor(Qt.ArrowCursor)
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
                self.resizeMode = ResizeMode.OTHER
            elif self.__correctionMode == CorrectionMode.MOVE:
                selectedBox.canvasPositionRatio = (selectedBox.pos().x() / self.width(), selectedBox.pos().y() / self.height())

            #TODO - Remove Invalid Bounding boxes with Area Threshold or Some Rules
            self.__correctionMode = CorrectionMode.OTHER

        super().mouseReleaseEvent(QMouseEvent)

    def resizeEvent(self, QResizeEvent):
        if QResizeEvent.oldSize().isValid():
            newSize = QResizeEvent.size()
            for box in self.__boxes:
                box.resize(newSize.width() * box.canvasBoxRatio[0], newSize.height() * box.canvasBoxRatio[1])
                box.move(newSize.width() * box.canvasPositionRatio[0], newSize.height() * box.canvasPositionRatio[1])

        super().resizeEvent(QResizeEvent)

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

    def __boundingBoxColor(self, label=None):
        if label is None:
            label = self.label

        color = QColor(self.colorTable[label])
        palette = QPalette()
        color.setAlpha(80)
        palette.setBrush(QPalette.Highlight, QBrush(color))

        return palette

    def removeBoundingBox(self, idx=None):
        if idx is None:
            idx = self.selectedIdx
            self.selectedIdx = -1

        if 0 <= idx < len(self.__boxes):
            self.__boxes[idx].hide()
            self.__boxes[idx].deleteLater()
            self.__boxes.pop(idx)
            self.changeBoxNum.emit(len(self.__boxes))

    def __getResizeDimension(self, box, mousePos, resizeMode):
        oldTopLeftX, oldTopLeftY = box.pos().x(), box.pos().y()
        oldBottomRightX, oldBottomRightY = oldTopLeftX + box.width(), oldTopLeftY + box.height()
        oldWidth, oldHeight = box.width(), box.height()

        mousePos = self.__clipCoordinateInWidget(mousePos)

        if resizeMode == ResizeMode.TOPLEFT:
            newX, newY = mousePos.x(), mousePos.y()
            newW, newH = oldBottomRightX - mousePos.x(), oldBottomRightY - mousePos.y()
        elif resizeMode == ResizeMode.TOP:
            newX, newY = oldTopLeftX, mousePos.y()
            newW, newH = oldWidth, oldBottomRightY - mousePos.y()
        elif resizeMode == ResizeMode.TOPRIGHT:
            newX, newY = oldTopLeftX, mousePos.y()
            newW, newH = mousePos.x() - oldTopLeftX, oldBottomRightY - mousePos.y()
        elif resizeMode == ResizeMode.RIGHT:
            newX, newY = oldTopLeftX, oldTopLeftY
            newW, newH = mousePos.x() - oldTopLeftX, oldHeight
        elif resizeMode == ResizeMode.RIGHTBOTTOM:
            newX, newY = oldTopLeftX, oldTopLeftY
            newW, newH = mousePos.x() - oldTopLeftX, mousePos.y() - oldTopLeftY
        elif resizeMode == ResizeMode.BOTTOM:
            newX, newY = oldTopLeftX, oldTopLeftY
            newW, newH = oldWidth, mousePos.y() - oldTopLeftY
        elif resizeMode == ResizeMode.BOTTOMLEFT:
            newX, newY = mousePos.x(), oldTopLeftY
            newW, newH = oldBottomRightX - mousePos.x(), mousePos.y() - oldTopLeftY
        elif resizeMode == ResizeMode.LEFT:
            newX, newY = mousePos.x(), oldTopLeftY
            newW, newH = oldBottomRightX - mousePos.x(), oldHeight

        return (newX, newY, newW, newH)

    def __changeCursor(self, cursorShape):
        QApplication.setOverrideCursor(QCursor(cursorShape))

    def __findResizingBox(self, QMouseEvent):
        for idx, box in enumerate(self.__boxes):
            resizeMode = self.__mouseOnEdge(box, QMouseEvent)
            if resizeMode != ResizeMode.OTHER:
                return idx, resizeMode
        return -1, ResizeMode.OTHER

    def __mouseOnEdge(self, box:BoundingBox, QMouseEvent):
        if box.pointOnTopLeft(QMouseEvent):
            self.__changeCursor(Qt.SizeFDiagCursor)
            return ResizeMode.TOPLEFT
        elif box.pointOnTop(QMouseEvent):
            self.__changeCursor(Qt.SizeVerCursor)
            return ResizeMode.TOP
        elif box.pointOnTopRight(QMouseEvent):
            self.__changeCursor(Qt.SizeBDiagCursor)
            return ResizeMode.TOPRIGHT
        elif box.pointOnRight(QMouseEvent):
            self.__changeCursor(Qt.SizeHorCursor)
            return ResizeMode.RIGHT
        elif box.pointOnRightBottom(QMouseEvent):
            self.__changeCursor(Qt.SizeFDiagCursor)
            return ResizeMode.RIGHTBOTTOM
        elif box.pointOnBottom(QMouseEvent):
            self.__changeCursor(Qt.SizeVerCursor)
            return ResizeMode.BOTTOM
        elif box.pointOnBottomLeft(QMouseEvent):
            self.__changeCursor(Qt.SizeBDiagCursor)
            return ResizeMode.BOTTOMLEFT
        elif box.pointOnLeft(QMouseEvent):
            self.__changeCursor(Qt.SizeHorCursor)
            return ResizeMode.LEFT
        else:
            self.__changeCursor(Qt.ArrowCursor)
            return ResizeMode.OTHER

    def __clipCoordinateInWidget(self, QMouseEvent):
        clipCoord = QPoint()
        clipCoord.setX(max(0, min(QMouseEvent.x(), self.width()-1)))
        clipCoord.setY(max(0, min(QMouseEvent.y(), self.height()-1)))

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
        self.windowWidth = 800
        self.windowHeight = 800
        self.windowTitle = AppString.TITLE.value
        self.windowXPos = 300
        self.windowYPos = 200

        self.allowDataType = '(*.jpg *.png, *.jpeg)'

    def setupUi(self):
        self.loadBtn = QAction(QIcon('./icon/plus-square-outline.svg'), AppString.LOAD.value, self)
        self.loadBtn.setIconText(AppString.LOAD.value)
        self.saveBtn = QAction(QIcon('./icon/download-outline.svg'), AppString.SAVE.value, self, shortcut="Ctrl+S")
        self.saveBtn.setIconText(AppString.SAVE.value)
        self.description = QAction(QIcon('./icon/question-mark-outline.svg'), AppString.DESCRIPTION.value, self)
        self.description.setIconText(AppString.DESCRIPTION.value)
        self.autoLabelBtn = QAction(QIcon('./icon/crop-outline.svg'), AppString.AUTOLABEL.value, self)
        self.autoLabelBtn.setIconText(AppString.AUTOLABEL.value)

        self.labelComboBox = QComboBox()

        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.setMovable(False)
        self.toolbar.addActions([self.loadBtn, self.saveBtn, self.description, self.autoLabelBtn])

        for action in self.toolbar.actions():
            widget = self.toolbar.widgetForAction(action)
            widget.setFixedSize(85, 60)

        self.toolbar.addWidget(self.labelComboBox)

        self.toolbar.setIconSize(QSize(30, 30))
        self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon | Qt.AlignLeading)
        self.toolbar.setStyleSheet('QToolBar{spacing:0px;}')

        self.viewer = Viewer(self)
        self.viewer.setScaledContents(True)
        self.viewer.setFocusPolicy(Qt.StrongFocus)

        self.notification = QLabel(Mode.LABELING.name, self)
        self.notification.setStyleSheet('QWidget { background-color: %s }' % QColor(0, 255, 0).name())
        self.boundingBoxNum = QLabel('Box: 0')

        self.bottomBar = self.statusBar()
        self.bottomBar.addWidget(self.notification)
        self.bottomBar.addWidget(self.boundingBoxNum)

        self.setCentralWidget(self.viewer)
        self.setGeometry(self.windowXPos, self.windowYPos, self.windowWidth, self.windowHeight)
        self.setWindowTitle(self.windowTitle)


class Labeling(QMainWindow, MainUI):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.loadBtn.triggered.connect(self.openFileDialogue)
        self.saveBtn.triggered.connect(self.saveFileDialogue)
        self.viewer.changeBoxNum.connect(self.changeBoxNum)

        for label in Label:
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(self.viewer.colorTable[label]))
            self.labelComboBox.addItem(QIcon(pixmap), label.value)

        self.labelComboBox.setCurrentIndex(len(Label)-1)
        self.labelComboBox.currentTextChanged.connect(self.viewer.setLabel)
        self.labelComboBox.setCurrentIndex(0)
        self.show()
        self.loadImage = None

    def initialize(self):
        self.labelComboBox.setCurrentIndex(0)
        self.changeBoxNum(0)

    def keyPressEvent(self, QKeyEvent):
        if not self.viewer.makeBoundingBox and self.viewer.correctionMode == CorrectionMode.OTHER:
            if QKeyEvent.key() == Qt.Key_I:
                self.viewer.mode = Mode.CORRECTION
            elif QKeyEvent.key() == Qt.Key_Escape:
                self.viewer.mode = Mode.LABELING
            self.__changeModeLabel(self.viewer.mode)

        if QKeyEvent.key() == Qt.Key_Delete:
            if self.viewer.mode == Mode.CORRECTION:
                self.viewer.removeBoundingBox()

        super().keyPressEvent(QKeyEvent)

    @pyqtSlot(int)
    def changeBoxNum(self, num):
        self.boundingBoxNum.setText('Box: {}'.format(num))

    def openFileDialogue(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Open', '', 'Image files {}'.format(self.allowDataType))

        if fileName != '':
            rawImage = QImage(fileName)
            self.initialize()
            self.viewer.initialize()
            self.loadImage = ImageContainer(rawImage, fileName.split('/')[-1])
            self.viewer.setPixmap(QPixmap.fromImage(rawImage.scaled(self.viewer.width(), self.viewer.height())))

    def saveFileDialogue(self):
        if self.loadImage is not None:
            xmlName = self.loadImage.fileName.split('.')[0] + '.xml'
            savePath, fileType = QFileDialog.getSaveFileName(self, 'Save', xmlName, 'xml files {}'.format('*.xml'))

            if savePath.split('/')[-1].split('.')[-1] != 'xml':
                savePath += '.xml'

            self.__saveToXml(savePath)

    def __saveToXml(self, filePath):
        bndBox = self.viewer.boxes
        annotation = xml_root(self.loadImage.fileName, self.loadImage.imageHeight, self.loadImage.imageWidth)

        instances = []

        for box in bndBox:
            xmin, ymin, width, height, label = box
            positionRatio = (xmin/self.viewer.width(), ymin/self.viewer.height())
            scaleRatio = (width/self.viewer.width(), height/self.viewer.height())
            instances.append({'bbox': [self.loadImage.imageWidth * positionRatio[0], self.loadImage.imageHeight * positionRatio[1],
                                       self.loadImage.imageWidth * scaleRatio[0], self.loadImage.imageHeight * scaleRatio[1]],
                              'category_id':label.value})

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Labeling()
    sys.exit(app.exec_())