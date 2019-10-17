# -*- coding=GBK -*-
import cv2 as cv
import numpy as np
filename2 = "H:/Ptest/0.bmp"
# ͼ���ֵ�� 0��ɫ 1��ɫ
# ȫ����ֵ
def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (500, 500), cv.INTER_AREA)
    cv.imshow("or", gray)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # ���ɷ�,ȫ������Ӧ��ֵ ����0�ɸ�Ϊ�������ֵ���������
    print("��ֵ��%s" % ret)
    binary = cv.resize(binary, (500, 500), cv.INTER_AREA)
    cv.imshow("da", binary)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)  # TRIANGLE��,��ȫ������Ӧ��ֵ, ����0�ɸ�Ϊ�������ֵ��������ã������ڵ�������
    print("��ֵ��%s" % ret)
    ret = cv.resize(ret, (500, 500), cv.INTER_AREA)
    cv.imshow("TRIANGLE", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)  # �Զ�����ֵΪ150,����150���ǰ�ɫ С�ڵ��Ǻ�ɫ
    print("��ֵ��%s" % ret)
    cv.imshow("self", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)  # �Զ�����ֵΪ150,����150���Ǻ�ɫ С�ڵ��ǰ�ɫ
    print("��ֵ��%s" % ret)
    cv.imshow("op", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TRUNC)  # �ض� ����150���Ǹ�Ϊ150  С��150�ı���
    print("��ֵ��%s" % ret)
    cv.imshow("duan1", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO)  # �ض� С��150���Ǹ�Ϊ150  ����150�ı���
    print("��ֵ��%s" % ret)
    cv.imshow("duan2", binary)


src = cv.imread(filename2)
threshold_image(src)
cv.waitKey(0)
cv.destroyAllWindows()