import cv2
import numpy as np
from math import *
import math
import os
import glob as gb

def is_inp(name):
    return (name[-4:] in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])
inp_path = 'H:/Ptest'
all_inps = os.listdir(inp_path)
all_inp = [i for i in all_inps if is_inp(i)]
flat = 0
for l in range(len(all_inp)):
    path_ = os.path.join(inp_path, all_inp[l])
    save=str(path_[-11:-4])
    print(save)
    result = cv2.imread(path_)
    test = cv2.imread("H:/Ptest/0.jpg")
    test = cv2.medianBlur(test, 5)
    result = cv2.medianBlur(result, 5)
    gray1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edge_input = cv2.Canny(gray1, 50, 150)
    edge_output = cv2.Canny(gray2, 50, 150)
    img_delta = cv2.absdiff(edge_output, edge_input)
    (_, cnts, _) = cv2.findContours(img_delta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.namedWindow("a", 0)
    cv2.imshow("a",img_delta)
    cv2.imshow("ahu", edge_output)
    cv2.imshow("hja", img_delta)
    rect = []
    area = []
    for j in range(len(cnts)):
        area.append(cv2.contourArea(cnts[j]))  # 存放轮廓所包围的面积
    b = sorted(enumerate(area), key=lambda x: x[1])  # 降序排列，保存原下s标
    if(b!=[]):
        for i in range(b.__len__() - 1, b.__len__())[::-1]:  # 有x个碎片，则b.__len__()-x
            if(b[i][0]>10):
                m = b[i][0]
                rect.append(cv2.minAreaRect(cnts[m]))  # 画出一个矩形，这个矩形能把一个轮廓全部包含，且面积最小   即得到最小的外界矩阵
                box = np.int0(cv2.boxPoints(rect[b.__len__() - i - 1]))
                cv2.drawContours(result, [box], -1, (0, 255, 0), 3)
                cv2.imwrite("%s.jpg"%save,result)
            else:
                continue
cv2.waitKey(0)

