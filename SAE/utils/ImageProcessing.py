#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import scipy.ndimage.morphology as morp
import numpy as np
import cv2
import sys

EQUALIZATION_TYPE_NONE = "none"
EQUALIZATION_TYPE_RGB = "rgb"
EQUALIZATION_TYPE_HSV = "hsv"
EQUALIZATION_TYPE_GRAY = "gray"
EQUALIZATION_TYPE_GR_CLAHE = "gr_clahe"
EQUALIZATION_TYPE_CO_CLAHE = "co_clahe"
EQUALIZATION_TYPE_HSV_CLAHE = "hsv_clahe"
EQUALIZATION_TYPE_WHITE_BALANCE = 'wb'

EQUALIZATION_TYPES = [
                EQUALIZATION_TYPE_NONE,
                EQUALIZATION_TYPE_RGB,
                EQUALIZATION_TYPE_HSV,
                EQUALIZATION_TYPE_GRAY,
                EQUALIZATION_TYPE_GR_CLAHE,
                EQUALIZATION_TYPE_CO_CLAHE,
                EQUALIZATION_TYPE_HSV_CLAHE,
                EQUALIZATION_TYPE_WHITE_BALANCE
]

def whiteBalanceForRGBImage(image, perc = 0.05):
    return np.dstack([whiteBalanceForChannelImage(channel, 0.05) for channel in cv2.split(image)] )

def whiteBalanceForChannelImage(channel, perc = 0.05):

    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel


# ----------------------------------------------------------------------------
def apply_equalization(img_x, type):
    if type == EQUALIZATION_TYPE_NONE:
        return img_x
    elif type == EQUALIZATION_TYPE_RGB:
        img_x = equalize_hist_rgb(img_x)
    elif type == EQUALIZATION_TYPE_HSV:
        img_x = equalize_hist_hsv(img_x)
    elif type == EQUALIZATION_TYPE_GRAY:
        img_x = equalize_hist_gray(img_x)
    elif type == EQUALIZATION_TYPE_GR_CLAHE:
        img_x = equalize_clahe_gray(img_x)
    elif type == EQUALIZATION_TYPE_CO_CLAHE:
        img_x = equalize_clahe_lab(img_x)
    elif type == EQUALIZATION_TYPE_HSV_CLAHE:
        img_x = equalize_clahe_hsv(img_x)
    elif type == EQUALIZATION_TYPE_WHITE_BALANCE:
        img_x = whiteBalanceForRGBImage(img_x)
    else:
        raise Exception('Undefined equalization type: ' + type)

    return img_x


# ----------------------------------------------------------------------------
def equalize_hist_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    return equ


# ----------------------------------------------------------------------------
# https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
def equalize_hist_rgb(img):
    channels = cv2.split(img)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))
    equ = cv2.merge(eq_channels)
    #equ = cv2.cvtColor(equ, cv2.COLOR_BGR2RGB)
    return equ


# ----------------------------------------------------------------------------
# https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
def equalize_hist_hsv(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    equ = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)  #cv2.COLOR_HSV2RGB)
    return equ


# ----------------------------------------------------------------------------
# https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
def equalize_clahe_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(gray)
    return equ


# ----------------------------------------------------------------------------
# https://stackoverflow.com/a/41075028
def equalize_clahe_lab(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)         # Apply CLAHE to L-channel
    equ = cv2.cvtColor(cv2.merge([cl,a,b]), cv2.COLOR_LAB2BGR)
    return equ


# ----------------------------------------------------------------------------
def equalize_clahe_hsv(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    eq_V = clahe.apply(V)           # Apply CLAHE to V-channel
    equ = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)  #cv2.COLOR_HSV2RGB)
    return equ
