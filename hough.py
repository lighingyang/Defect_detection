import cv2 as cv
import numpy as np
filename1 = "H:/picture/0.jpg"
filename2 = "H:/picture/Ptest/"
str = "0.bmp"
image = cv.imread(filename2+str)
gay_img = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
img = cv.medianBlur(gay_img, 7)  # 进行中值模糊，去噪点
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=900, maxRadius=1000)

circles = np.uint16(np.around(circles))
print(circles)
num = 0
for i in circles[0, :]:  # 遍历矩阵每一行的数据
    cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
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

