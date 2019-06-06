import cv2
import numpy as np
import os

cam = cv2.VideoCapture(1)
kernel = np.ones((12,12),np.uint8)


def finddifference(img1,img2):
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    gapimg = cv2.absdiff(img1,img2)
    gapnum = cv2.countNonZero(gapimg)
    return gapnum

def loaddata():
    datanames = []
    dataimages = []

    Docs = os.listdir("Data/")
    for Doc in Docs:
            datanames.append(Doc.replace(".jpg",""))
            dataimages.append(cv2.imread("Data/"+Doc,0))
    return datanames,dataimages

def classify(image,datanames,dataimages):
    minindex = 0
    minval = finddifference(image,dataimages[0])
    for t in range(len(datanames)):
        gapval = finddifference(image,dataimages[t])
        if(gapval<minval):
            minval = gapval
            minindex = t
    return datanames[minindex]

datanames,dataimages = loaddata()

while True:
    ret, square = cam.read()
    cutsquare = square[0:200,0:250]
    cutsquarehsv = cv2.cvtColor(cutsquare,cv2.COLOR_BGR2HSV)
    minvals = np.array([0, 55, 70])
    maxvals = np.array([55, 255, 255])
    colorfilterresult = cv2.inRange(cutsquarehsv,minvals,maxvals)
    colorfilterresult = cv2.morphologyEx(colorfilterresult,cv2.MORPH_CLOSE, kernel)
    colorfilterresult = cv2.dilate(colorfilterresult,kernel, iterations=1)

    cnts, hierarchy = cv2.findContours(colorfilterresult, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    result = cutsquare.copy()
    maxw = 0
    maxh = 0
    maxindex = -1

    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)

        if(w>maxw and h>maxh):
            maxh = h
            maxw = w
            maxindex = t

    if(len(cnts) > 0):
        x,y,w,h  = cv2.boundingRect(cnts[maxindex])
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        handimg = colorfilterresult[y:y+h,x:x+w]
        print(classify(handimg,datanames,dataimages))

    cv2.imshow("square",square)
    cv2.imshow("cutsquare",cutsquare)
    cv2.imshow("colorfilterresult",colorfilterresult)
    cv2.imshow("result",result)


    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
