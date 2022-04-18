from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

import itertools

rng.seed(12345)

def getBoundingBoxes(image, val=100):
    threshold = val
    minContourSize= int(image.shape[0]*image.shape[1]*0.0025)
    
    img = np.copy(image)
    ROWS = img.shape[0]
    COLS = img.shape[1]

    for j in range(COLS):
        img[0, j] = 0
        img[1, j] = 0
        img[2, j] = 0

        img[ROWS-1, j] = 0
        img[ROWS-2, j] = 0
        img[ROWS-3, j] = 0
    
    for i in range(ROWS):
        img[i, 0] = 0
        img[i, 1] = 0
        img[i, 2] = 0

        img[i, COLS-1] = 0
        img[i, COLS-2] = 0
        img[i, COLS-3] = 0
    
    
    
    #minContourSize = 500
    im = np.uint8(img)
    canny_output = cv.Canny(im, threshold, threshold * 2)
    
    contours, herarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []
    boundRect = []

    # calculate points for each contour
    for i in range(len(contours)):

        contour_i = contours[i]
        contour_poly = cv.approxPolyDP(contour_i, 3, True)
        hull_i = cv.convexHull(contour_poly, False)

        area = cv.contourArea(hull_i)

        if (area > minContourSize):
            bbox_i = cv.boundingRect(hull_i)

            rect_by_corners = (bbox_i[1], bbox_i[0], bbox_i[1]+bbox_i[3], bbox_i[0]+bbox_i[2])

            boundRect.append(rect_by_corners)

        #if (cv.contourArea(hull_i) > 0):
            

    return boundRect


