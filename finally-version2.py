import cv2 as cv
import numpy as np
import os


filename1 = "F:/picture/0.jpg"

'''
样本照片储存的文件夹
'''
filename2 = "F:/picture/Ptest/"

'''
标准样图片 str = 0.bmp
'''
str = "0.bmp"

def abs(edge_input,edge_output,result):                    #两张图片查分
    img_delta = cv.absdiff(edge_input, edge_output)
    cv.namedWindow("result", 0)
    cv.imshow("result", img_delta)
    cv.waitKey(0)
    (cnts, _) = cv.findContours(img_delta, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
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
            if (b[i][0] > 1):
                m = b[i][0]
                rect.append(cv.minAreaRect(cnts[m]))  # 画出一个矩形，这个矩形能把一个轮廓全部包含，且面积最小   即得到最小的外界矩阵
                box = np.int0(cv.boxPoints(rect[b.__len__() - i - 1]))
                cv.drawContours(result, [box], -1, (0, 255, 0), 3)
                # break
            else:
                continue
    # cv2.drawContours(result,cnts,-1,(0,0,255),3)
    # result = cv2.resize(result,(500,500),cv2.INTER_AREA)
    cv.namedWindow("result", 0)
    cv.imshow("result", result)
    # cv2.imwrite("ans.jpg",result)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
霍夫圆检测函数
输入：image：需要霍夫圆检测的图像
      res_r:检测时的圆半径
'''
def hough(image,res_r):                                                #霍夫变换函数
    gay_img = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    img = cv.medianBlur(gay_img, 7)  # 进行中值模糊，去噪点
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=900, maxRadius=1000)
    circles = np.uint16(np.around(circles))
    #print(circles)
    num = 0
    for i in circles[0, :]:  # 遍历矩阵每一行的数据
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)    #画出霍夫圆
        cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        num += 1 #圆的数量，即检测出原片的数量
    #print("num  =", num)
    cir = circles[0][0]
    cir_c = cir[0]
    cir_r = cir[1]
    r = cir[2] - 89
    if res_r != 0:
        r = res_r
    if num == 1:
        img = image[cir_r - r:cir_r + r, cir_c - r:cir_c + r]
        row, col = img.shape[0:2]
        cir_r = r
        cir_c = r
        # for i in range(row):
        #     for j in range(col):
        #         if(( i - cir_r )**2 + ( j - cir_c )**2 > r**2):
        #             img[i][j][0] = 0
        #             img[i][j][1] = 0
        #             img[i][j][2] = 0
        #         if ((i - cir_r) ** 2 + (j - cir_c) ** 2 < 180000):
        #             img[i][j][0] = 0
        #             img[i][j][1] = 0
        #             img[i][j][2] = 0
        cv.namedWindow("convert", 0)
        cv.imshow("convert", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return r, img

'''
边缘检测函数
参数：image:输入的需要处理的图像
输出：image：边缘检测后的二值图像
'''
def B_solve(image):                           #边缘检测
    image = cv.medianBlur(image, 5)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.Canny(image, 50, 60)
    return image

if __name__ == "__main__":
    res_r = 0
    res = cv.imread(filename2 + str)
    res_r, res = hough(res, res_r)
    res = B_solve(res)
    cv.namedWindow("test", 0)
    cv.imshow("test",res)
    cv.waitKey(0)
    images = os.listdir(filename2)
    for str1 in (images):
        if str1 == str:
            continue
        else:
            print(str1)
            test1 = cv.imread(filename2+str1)
            test_r, test2 = hough(test1, res_r)
            test3 = B_solve(test2)
            cv.namedWindow("1", 0)
            cv.namedWindow("2", 0)
            cv.imshow("1", res)
            cv.imshow("2", test3)
            cv.waitKey(0)
            abs(res, test3, test2)