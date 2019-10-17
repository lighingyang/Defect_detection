import cv2 as cv
filename = "H:/picture/"
img = cv.imread(filename+'1.jpg')
print(img)
cv.namedWindow("res",0)
cv.imshow("res",img)
cv.waitKey(0)
cv.destroyAllWindows()
row ,col = img.shape[0:2]
for i in range(row):
    for j in range(col):
        img[i][j][0] = 0
        img[i][j][1] = 0
        img[i][j][2] = 0
cv.imshow("res",img)
cv.waitKey(0)
cv.destroyAllWindows()