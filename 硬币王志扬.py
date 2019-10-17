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
filename2 = "H:/picture/Ptest/"
def solve(test,str2):
    #test= panelAbstract(cv2.imread(filename1 + str1))
    #result= panelAbstract(cv2.imread(filename1 + str2))
    result = cv2.imread(filename2 + str2)
    left, right = fun_col(result)
    up, down = fun_row(result)
    result = result[up:down,left:right]
    row_1,col_1 = test.shape[0:2]
    row_2,col_2 = result.shape[0:2]
    row = max(row_1,row_2)
    col = max(col_1,col_2)
    test = cv2.resize(test, (row, col), cv2.INTER_AREA)
    result = cv2.resize(result, (row, col), cv2.INTER_AREA)
    cv2.namedWindow("pre", 0)
    cv2.namedWindow("now", 0)
    #w,h = test.shape[0:2]
    #point1 = np.array([[308, 230], [500, 230], [308, 640], [500, 640]], dtype="float32")
    #point2 = np.array([[308, 230], [500, 230], [155, 30], [835, 30]], dtype="float32")
    #M = cv2.getPerspectiveTransform(point1, point2)
    #out_img = cv2.warpPerspective(test , M, (w, h))
    #cv2.imshow("111",out_img)
    #cv2.waitKey(0)
    #res = cv2.resize(test,(500,500),cv2.INTER_AREA)
    test = cv2.medianBlur(test, 5)
    result = cv2.medianBlur(result, 5)
    #test = cv2.blur(test, (5, 5))
    #result = cv2.blur(result, (5, 5))
    #test = cv2.resize(test,(500,500),cv2.INTER_AREA)
    #result = cv2.resize(result,(500,500),cv2.INTER_AREA)
    #test = cv2.GaussianBlur(test,(5,5),0)
    #result = cv2.GaussianBlur(result,(5,5),0)
    gray1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imshow("pre", test)
    cv2.imshow("now", result)
    cv2.waitKey(0)
    #kerne = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #gray1 = cv2.erode(gray1, None, iterations=4)
    #gray1 = cv2.dilate(gray1, None, iterations=4)
    #gray2 = cv2.erode(gray2, None, iterations=4)
    #gray2 = cv2.dilate(gray2, None, iterations=4)
    #edge_input = cv2.(gray2, cv2.CV_8U, 1, 0,  ksize =3 )
    #edge_input = cv2.resize(edge_input,(500,500),cv2.INTER_AREA)
    edge_input = cv2.Canny(gray1, 50, 150)
    edge_output = cv2.Canny(gray2, 50, 150)
    #edge_output = cv2.resize(edge_output,(500,500),cv2.INTER_AREA)
    cv2.namedWindow("111",0)
    cv2.namedWindow("222",0)
    cv2.imshow("111",edge_input)
    cv2.imshow("222",edge_output)
    cv2.waitKey(0)
    img_delta = cv2.absdiff(edge_input, edge_output)
    cv2.namedWindow("result",0)
    cv2.imshow("result",img_delta)
    cv2.waitKey(0)
    ( cnts, _)  = cv2.findContours(img_delta, cv2.RETR_TREE,  cv2.CHAIN_APPROX_TC89_L1)
    #print(type(cnts))
    #print(type(cnts[0]))
    rect = []
    area = []
    for j in range(len(cnts)):
        area.append(cv2.contourArea(cnts[j]))  # 存放轮廓所包围的面积
    b = sorted(enumerate(area), key=lambda x: x[1])  # 降序排列，保存原下s标
    #print(type(b))
    if b != []:
        for i in range(b.__len__() - 1, b.__len__())[::-1]:  # 有x个碎片，则b.__len__()-x
            if (b[i][0] > 10):
                m = b[i][0]
                rect.append(cv2.minAreaRect(cnts[m]))  # 画出一个矩形，这个矩形能把一个轮廓全部包含，且面积最小   即得到最小的外界矩阵
                box = np.int0(cv2.boxPoints(rect[b.__len__() - i - 1]))
                cv2.drawContours(result, [box], -1, (0, 255, 0), 3)
                #break
            else:
                continue
    #cv2.drawContours(result,cnts,-1,(0,0,255),3)
    #result = cv2.resize(result,(500,500),cv2.INTER_AREA)
    cv2.namedWindow("result",0)
    cv2.imshow("result",result)
    #cv2.imwrite("ans.jpg",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def panelAbstract(srcImage):
    #     #   read pic shape
    imgHeight,imgWidth = srcImage.shape[:2]
    print(srcImage.shape[:2])
    imgWidth = int(imgWidth)
    imgHeight = int(imgHeight)
    # 均值聚类提取前景:二维转一维
    imgVec = np.float32(srcImage.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret,label,clusCenter = cv2.kmeans(imgVec,2,None,criteria,10,flags)
    clusCenter = np.uint8(clusCenter)
    clusResult = clusCenter[label.flatten()]
    imgres = clusResult.reshape((srcImage.shape))
    imgres = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)
    bwThresh = int((np.max(imgres)+np.min(imgres))/2)
    _,thresh = cv2.thr
    threshRotate = cv2.merge([thresh,thresh,thresh])
    #find contours
    (_,contours,_) = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    minvalx = np.max([imgHeight,imgWidth]);maxvalx = 0
    minvaly = np.max([imgHeight,imgWidth]);maxvaly = 0
    maxconArea = 0;maxAreaPos = -1
    for i in range(len(contours)):
        if maxconArea < cv2.contourArea(contours[i]):
            maxconArea = cv2.contourArea(contours[i])
            maxAreaPos = i
    objCont = contours[maxAreaPos]
    # 旋转校正前景
    rect = cv2.minAreaRect(objCont)
    for j in range(len(objCont)):
        minvaly = np.min([minvaly,objCont[j][0][0]])
        maxvaly = np.max([maxvaly,objCont[j][0][0]])
        minvalx = np.min([minvalx,objCont[j][0][1]])
        maxvalx = np.max([maxvalx,objCont[j][0][1]])
    if rect[2] <=-45:
        rotAgl = 90 +rect[2]
    else:
        rotAgl = rect[2]
    if rotAgl == 0:
        panelImg = srcImage[minvalx:maxvalx,minvaly:maxvaly,:]
    else:
        rotCtr = rect[0]
        rotCtr = (int(rotCtr[0]),int(rotCtr[1]))
        rotMdl = cv2.getRotationMatrix2D(rotCtr,rotAgl,1)
        imgHeight,imgWidth = srcImage.shape[:2]
        #图像的旋转
        dstHeight = math.sqrt(imgWidth *imgWidth + imgHeight*imgHeight)
        dstRotimg = cv2.warpAffine(threshRotate,rotMdl,(int(dstHeight),int(dstHeight)))
        dstImage = cv2.warpAffine(srcImage,rotMdl,(int(dstHeight),int(dstHeight)))
        dstRotimg = cv2.cvtColor(dstRotimg,cv2.COLOR_BGR2GRAY)
        _,dstRotBW = cv2.threshold(dstRotimg,127,255,0)
        imgCnt,contours, hierarchy = cv2.findContours(dstRotBW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0;maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x,y,w,h = cv2.boundingRect(contours[maxAreaPos])
        #提取前景：panel
        panelImg = dstImage[int(y):int(y+h),int(x):int(x+w),:]

    return panelImg
def fun_col(test):
    #test = cv2.imread(filename2+str)
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
def fun_row(test):
    #test = cv2.imread(filename2+str)
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
if __name__ == "__main__":
    str1 = "0.bmp"
    test = cv2.imread(filename2 + str1)
    left, right = fun_col(test)
    up, down = fun_row(test)
    roi_test = test[up:down,left:right]
    imgs = os.listdir(filename2)
    for str2 in (imgs):
        if str2 == str1:
            continue
        else:
            solve(roi_test,str2)