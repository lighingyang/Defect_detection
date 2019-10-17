import argparse
import datetime
#import imutils
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from math import *
import math
filename1 = "H:/picture/"
filename2 = "H:/Ptest/"
def fun_col(jpg):
    test = cv2.imread(filename2 + jpg)
    GrayImage = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)  # 将BGR图转为灰度图
    ret, thresh1 = cv2.threshold(GrayImage, 130, 255, cv2.THRESH_BINARY)  # 将图片进行二值化（130,255）之间的点均变为255（背景）
    # print(thresh1[0,0])#250  输出[0,0]这个点的像素值  				#返回值ret为阈值
    # print(ret)#130
    (h, w) = thresh1.shape  # 返回高和宽
    # print(h,w)#s输出高和宽
    a = [0 for z in range(0, w)]
    #print(a)  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if thresh1[i, j] == 0:  # 如果改点为黑点
                a[j] += 1  # 该列的计数器加一计数
                thresh1[i, j] = 255  # 记录完后将其变为白色
        # print (j)

    #
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i, j] = 0  # 涂黑
    # 此时的thresh1便是一张图像向垂直方向上投影的直方图
    # 如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息

    # img2 =Image.open('0002.jpg')
    # img2.convert('L')
    # img_1 = np.array(img2)
    plt.imshow(thresh1, cmap=plt.gray())
    plt.show()
    cv2.namedWindow("img",0)
    cv2.imshow('img', thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    num = 1
    for i in a:
        print(num,':',i)
        num+=1
    left = 0xcfcfcf ; right = 0xcfcfcf
    for i in range(1000,2000):
        left = min(left,a[i])
    for i in range(3000,4000):
        right = min(right,a[i])
    idx1 = 1000
    while idx1 <=2000:
        if a[idx1]==left:
            break
        idx1+=1
    idx2 = 3000
    while idx2 <=4000:
        if a[idx2]==right:
            break
        idx2+=1
    print(idx1,'     ',idx2)
    return idx1,idx2
def fun_row(jpg):
    test = cv2.imread(filename2 + jpg)
    GrayImage = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)  # 将BGR图转为灰度图
    ret, thresh1 = cv2.threshold(GrayImage, 130, 255, cv2.THRESH_BINARY)  # 将图片进行二值化（130,255）之间的点均变为255（背景）
    # print(thresh1[0,0])#250  输出[0,0]这个点的像素值  				#返回值ret为阈值
    # print(ret)#130
    (h, w) = thresh1.shape  # 返回高和宽
    # print(h,w)#s输出高和宽
    a = [0 for z in range(0, h)]
    # print(a)  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # 记录每一列的波峰
    for i in range(0, h):  # 遍历一行
        for j in range(0, w):  # 遍历一列
            if thresh1[i, j] == 0:  # 如果改点为黑点
                a[i] += 1  # 该列的计数器加一计数
                thresh1[i, j] = 255  # 记录完后将其变为白色
        # print (j)

    #
    for i in range(0, h):  # 遍历每一行
        for j in range(0, (a[i])):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i, j] = 0  # 涂黑
    # 此时的thresh1便是一张图像向垂直方向上投影的直方图
    # 如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息

    # img2 =Image.open('0002.jpg')
    # img2.convert('L')
    # img_1 = np.array(img2)
    plt.imshow(thresh1, cmap=plt.gray())
    plt.show()
    cv2.namedWindow("img", 0)
    cv2.imshow('img', thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    num = 1
    for i in a:
        print(num, ':', i)
        num += 1
    up = 0xcfcfcf
    down = 0xcfcfcf
    for i in range(250, 1000):
        up = min(up, a[i])
    for i in range(2000, 3000):
        down = min(down, a[i])
    idx1 = 250
    while idx1 <= 2000:
        if a[idx1] == up:
            break
        idx1 += 1
    idx2 = 2000
    while idx2 <= 3000:
        if a[idx2] == down:
            break
        idx2 += 1
    print(idx1, '     ', idx2)
    return idx1,idx2
def fun(img):
    cv2.imshow("1",img)
    cv2.waitKey(0)
if __name__ == "__main__":
    '''
    left,right =fun_col("0.bmp")
    up,down = fun_row("0.bmp")
    img = cv2.imread(filename2 + '0.bmp')
    roi = img[up:down,left:right]
    cv2.namedWindow("roi",0)
    cv2.imshow("roi",roi)
    cv2.waitKey(0)
    '''
    v = [1,2,3,4]
    vv = v[0:2]
    print(vv)