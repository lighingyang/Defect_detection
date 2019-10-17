# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qtfinally.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
import sys
import cv2 as cv
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog
from PyQt5 import QtCore, QtGui, QtWidgets
filename1 = "H:/picture/0.jpg"
filename2 = "E:/images4/"
str = "01.bmp"
def abs(edge_input,edge_output,result):                    #两张图片查分
    img_delta = cv.absdiff(edge_input, edge_output)
    cv.namedWindow("result", 0)
    cv.imshow("result", img_delta)
    cv.waitKey(0)
    (_,
     cnts, _) = cv.findContours(img_delta, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
    # print(type(cnts))
    # print(type(cnts[0]))
    rect = []
    area = []
    for j in range(len(cnts)):
        area.append(cv.contourArea(cnts[j]))  # 存放轮廓所包围的面积
    b = sorted(enumerate(area), key=lambda x: x[1])  # 降序排列，保存原下s标
    # print(type(b))
    if b != []:
        for i in range(b.__len__() - 1, b.__len__())[::-1]:  # 有x个碎片，则b.__len__()-x
            if (b[i][0] > 1

            ):
                m = b[i][0]
                rect.append(cv.minAreaRect(cnts[m]))  # 画出一个矩形，这个矩形能把一个轮廓全部包含，且面积最小   即得到最小的外界矩阵
                box = np.int0(cv.boxPoints(rect[b.__len__() - i - 1]))
                cv.drawContours(result, [box], -1, (0, 255, 0), 3)
                # break
            else:
                continue
    # cv2.drawContours(result,cnts,-1,(0,0,255),3)
    #result = cv2.resize(result,(500,500),cv2.INTER_AREA)
    cv.namedWindow("result", 0)
    cv.imshow("result", result)
    # cv2.imwrite("ans.jpg",result)
    cv.waitKey(0)
    cv.destroyAllWindows()
def hough(image):                                                #霍夫变换函数
    gay_img = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    img = cv.medianBlur(gay_img, 7)  # 进行中值模糊，去噪点
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=900, maxRadius=1000)

    circles = np.uint16(np.around(circles))
    print(circles)
    num = 0
    for i in circles[0, :]:  # 遍历矩阵每一行的数据
        #cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)    #画出霍夫圆
        #cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        num += 1
    cir = circles[0][0]
    cir_c = cir[0]
    cir_r = cir[1]
    r = cir[2]
    if num == 1:
        img = image[cir_r - r:cir_r + r, cir_c - r:cir_c + r]
        cv.namedWindow("convert", 0)
        cv.imshow("convert", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return img
def B_solve(image):                           #边缘检测
    image = cv.medianBlur(image, 5)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.Canny(image, 50, 60)
    return image

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(260, 90, 211, 51))
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(110, 150, 581, 31))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 220, 161, 41))
        self.label_2.setObjectName("label_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(110, 260, 581, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(290, 350, 161, 31))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.pushButton.clicked.connect(self.slot1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "请输入标准图片所在位置"))
        self.label_2.setText(_translate("MainWindow", "请输入样本文件夹所在位置"))
        self.pushButton.setText(_translate("MainWindow", "开始检测"))

    def slot1(self):
        ans=self.textEdit.toPlainText()
        an=self.textEdit_2.toPlainText()
        res = cv.imread(ans)
        res = hough(res)
        res = B_solve(res)
        images = os.listdir(an)
        for str1 in (images):
            if str1 == str:
                continue
            else:
                print(str1)
                test1 = cv.imread(ans)
                test2 = hough(test1)
                test3 = B_solve(test2)
                cv.namedWindow("1",0)
                cv.namedWindow("2", 0)
                cv.imshow("1",res)
                cv.imshow("2", test3)
                cv.waitKey(0)
                abs(res,test3,test2)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
