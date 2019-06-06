import cv2
import numpy as np
import os

cam = cv2.VideoCapture(1)
kernel = np.ones((12,12),np.uint8)
name = "one"



while True:

    ret, square = cam.read()
    cutsquare = square[0:200, 0:250]
    cutsquarehsv = cv2.cvtColor(cutsquare, cv2.COLOR_BGR2HSV)
    minvals = np.array([0, 55, 70])
    maxvals = np.array([55, 255, 255])
    colorfilterresult = cv2.inRange(cutsquarehsv, minvals, maxvals)
    colorfilterresult = cv2.morphologyEx(colorfilterresult, cv2.MORPH_CLOSE, kernel)
    colorfilterresult = cv2.dilate(colorfilterresult, kernel, iterations=1)

    cnts, hierarchy = cv2.findContours(colorfilterresult, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    result = cutsquare.copy()
    maxw = 0
    maxh = 0
    maxindex = -1

    for t in range(len(cnts)):
        cnt = cnts[t]
        x, y, w, h = cv2.boundingRect(cnt)

        if (w > maxw and h > maxh):
            maxh = h
            maxw = w
            maxindex = t

    if (len(cnts) > 0):
        x, y, w, h = cv2.boundingRect(cnts[maxindex])
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        handimg = colorfilterresult[y:y + h, x:x + w]


    cv2.imshow("square", square)
    cv2.imshow("cutsquare", cutsquare)
    cv2.imshow("colorfilterresult", colorfilterresult)
    cv2.imshow("result", result)

cv2.imwrite("Data/"+name+".jpg",El_Resim)

cam.release()
cv2.destroyAllWindows()


