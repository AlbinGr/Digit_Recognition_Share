import sys
from PIL import Image
import tensorflow as tf
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QPen, QTransform, QFont


class Window(QMainWindow):
    release = False

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.drawing = False
        self.lastPoint = QPoint()
        self.image = QPixmap(600, 300)
        self.image.fill(color=Qt.white)
        self.setGeometry(100, 100, 600, 300)
        self.resize(self.image.width(), self.image.height())
        self.board = QLabel(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.conversion)

        self.clear_button = QPushButton(text="Clear board", parent=self)
        self.clear_button.setFixedSize(100, 60)
        self.clear_button.setObjectName("clear_button")
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.click

        self.display = QLabel(text="", parent=self)
        self.display.setFont(QFont('Arial', 72))

        self.show()

    def paintEvent(self, event):
        drawing_place = QRect()
        drawing_place.setRect(int(0.05 * self.height()), int(0.05 * self.height()), int(self.width() / 2.1),
                              int(self.height() / 1.1))
        self.board.setGeometry(drawing_place.x(), drawing_place.y(), drawing_place.width(), drawing_place.height())
        self.image = self.image.scaled(drawing_place.width(), drawing_place.height(),
                                       transformMode=Qt.SmoothTransformation)

        self.clear_button.move(drawing_place.x() + drawing_place.width() + 10, drawing_place.y())

        self.display.setGeometry(self.clear_button.x(),
                         self.clear_button.y() + self.clear_button.height() + 10, 100, 100)

        transform = QTransform()
        transform.translate(drawing_place.x(), drawing_place.y())
        self.image = self.image.transformed(transform)

        self.board.setPixmap(self.image)
        painter = QPainter(self.board)
        painter.drawPixmap(drawing_place, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            self.release = False
            self.timer.stop()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.translate(-int(0.05 * self.height()), -int(0.05 * self.height()))
            painter.setPen(QPen(Qt.black, int(self.height()/20), Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drawing = False

        self.release = True
        self.timer.start(2000)

    def conversion(self):
        """converting and displaying the recognition using self.model"""
        self.timer.stop()

        array = self.QPixmapToArray(self.image)
        input_image = self.SizeConversion(array)
        x = np.reshape(input_image, (1,28,28))

        output_result = self.model.predict(x, verbose=0)
        output_result = tf.nn.softmax(output_result)
        self.display.setText(str(np.argmax(output_result)))

    """Take in a QPixmap and return a np.array representing the image"""
    @staticmethod
    def QPixmapToArray(pixmap):
        # Get the size of the current pixmap
        channels_count = 3
        size = pixmap.size()
        h = size.width()
        w = size.height()

        # Get the QImage Item and convert it to a byte string
        qimg = pixmap.toImage()
        qimg.save("Files/image.jpg")

        im = Image.open("Files/image.jpg")
        arr = np.array(im)
        return arr

    """Method that convert an arbitrary sized image to a 28*28 grey scaled normalized image"""
    @staticmethod
    def SizeConversion(arr):

        result = np.zeros((28, 28))
        M = 0
        m = 255
        dy = int(len(arr)/28)
        dx = int(len(arr[0])/28)

        arr_grey = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        for i in range(28):
            for j in range(28):
                result[i][j] = arr_grey[dx*i:dx*(i+1), dy*j:dy*(j+1)].mean()
                if result[i][j] > M:
                    M = result[i][j]
                if m > result[i][j]:
                    m = result[i][j]

        result = 255-(result-m)/(M-m)*255

        im = Image.fromarray(result.reshape((28, 28)).astype('uint8'))
        im.save("Files/image28x28.jpg")

        return result

    def clear(self):
        self.image.fill(color=Qt.white)




