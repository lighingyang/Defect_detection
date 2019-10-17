import os
import cv2
import numpy as np
orginName = "orgin.jpg"                    #标准待测图像储存名称及格式格式
def preSolve():
    str = "H:/picture/"                    #检测图像存放的目录
    filenames = os.listdir(str)
    return filenames                       #把所有图象的全名以返回

def method_solve(now_img,orgin_img):       #图像处理函数
    '对两张图片进行处理'
    if 1 == 1:
        return now_img,0    #如果无缺陷 返回原图像和1
    else:
        solved_img = now_img.copy()
        return solve_img,1  #如果有缺陷，返回处理之后的图像和0
def solve(filenames):            #处理函数
    orgin_img = cv2.imread('H:/picture/' + orginName)
    for str in filenames:
        if os.path.isdir(str) or str == orginName: #如果是目录或者标准图像，查看下一张图像
            continue
        else :
            now_img =cv2.imread('H:/picture/' + str)
            solved_img ,i = method_solve(now_img, orgin_img)
            if i==0:  #如果无缺陷，查看下一张图片
                continue
            else:     #否则有缺陷，存储图片
                cv2.imwrite('H:/pictureSaving/'+str,solved_img) #

filenames = preSolve()
orgin_img = cv2.imread('H:/picture/' + orginName)
cv2.namedWindow('orginName',cv2.WINDOW_NORMAL)
cv2.imshow('orginName',orgin_img)
cv2.waitKey(0)
solve(filenames)
cv2.destroyAllWindows()