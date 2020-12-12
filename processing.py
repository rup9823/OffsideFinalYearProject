import numpy as np
import argparse
import cv2
import math
from collections import deque

image = None
original = None
roi = None
roi2, roi2_init = None,None

kernel = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 0, 0]],dtype=np.uint8)

ix,iy = 0,0
draw = False
rad_thresh = 15

def selectROI(event, x, y, flag, param):
    global image, ix, iy, draw, original, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            image = cv2.rectangle(original.copy(), (ix, iy), (x, y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        if draw:
            x1 = max(x, ix)
            y1 = max(y, iy)
            ix = min(x, ix)
            iy = min(y, iy)
            roi = original[iy:y1, ix:x1]
        draw = False

def getROIvid(frame, winName = 'input'):
    global image, original, roi
    roi = None
    image = frame.copy()
    original = frame.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, image)
        if roi is not None:
            cv2.destroyWindow(winName)
            return roi

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(winName)
            break

    return roi

def getLimits(roi):
    limits = None
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi)
    limits = [(int(np.amax(h)), int(np.amax(s)), 255), (int(np.amin(h)), int(np.amin(s)), int(np.amin(v)))]
    return limits

def applyMorphTransforms(mask):
    global kernel
    lower = 100
    upper = 255

    #mask = cv2.inRange(mask, lower, upper)
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, np.ones((5, 5)))

    return mask

def resize(image,width=400.0):
    r = float(width) / image.shape[0]
    dim = (int(image.shape[1] * r), int(width))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

def getHistogram(frame):
    roi_hist_A, roi_hist_B = None, None

    if roi_hist_A is None:
        roi = getROIvid(frame,'input team A')
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        roi_hist_A = cv2.calcHist([roi],[0,1],None,[180,256],[0,180,0,256])
        roi_hist_A = cv2.normalize(roi_hist_A, roi_hist_A, 0, 255, cv2.NORM_MINMAX)

    if roi_hist_B is None:
        roi = getROIvid(frame, 'input team B')
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist_B = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        roi_hist_B = cv2.normalize(roi_hist_B, roi_hist_B, 0, 255, cv2.NORM_MINMAX)

    return roi_hist_A, roi_hist_B
def check():
    if roi is not None:
        print('Offside')
    if roi2 is None:
        print('Not Offside')
def applyMorphTransforms2(backProj):
    global kernel
    lower = 50
    upper = 255
    mask = cv2.inRange(backProj, lower, upper)
    for i in range(0,2):
        mask = cv2.dilate(mask, kernel)
    for i in range(0,2):
        mask = cv2.erode(mask, np.ones((3, 3)))
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    return mask



def detectBallThresh(frame,limits):
    global rad_thresh
    upper = limits[0]
    lower = limits[1]
    center = None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = applyMorphTransforms(mask)
    #cv2.imshow('mask', mask)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    flag = False
    i=0
    if len(contours) > 0:
        for i in range(len(contours)):
            (_, radius) = cv2.minEnclosingCircle(contours[i])
            if radius < rad_thresh and radius > 8:
                flag = True
                break
        if not flag:
            return None, None

        M = cv2.moments(contours[i])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if(center[1]>173):
            #print(center)
            return center, contours[i]
    else:
        return None, None
    return None, None


def removeBG(frame, fgbg):
    bg_mask = fgbg.apply(frame)
    bg_mask = cv2.dilate(bg_mask, np.ones((5, 5)))
    frame = cv2.bitwise_and(frame, frame, mask=bg_mask)
    return frame
