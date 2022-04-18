#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#==============================================================================
"""
Created on Tue Sep  4 21:06:49 2018

@author: Francisco J. Castellanos
@project name: DAMA
"""
#==============================================================================


import cv2
import numpy as np


def showImage(img, title = "Image", destroy_with_any_key=False):
    
    cv2.imshow(title,img)
    
    if (destroy_with_any_key):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    





def scaleImage(img, scale, interpolation = cv2.INTER_LINEAR):
    
    newx,newy = int(img.shape[1]*scale),int(img.shape[0]*scale) #new size (w,h)
    img2 = img.copy()
    return cv2.resize(img2,(newx,newy), interpolation=interpolation)


def redimImage(img, height, width, interpolation = cv2.INTER_LINEAR):
    
    img2 = img.copy()
    return cv2.resize(img2,(width,height), interpolation=interpolation)


def concatenateImage(img1, img2, vertically):
    if (vertically):
        return np.concatenate((img1, img2), axis=0)
    else:
        return np.concatenate((img1, img2), axis=1)



#==============================================================================
# Code Tests...
#==============================================================================

if __name__ == "__main__":
    
    print ("Main")
    
    #img = FileManager.loadImage ("tests/src", "e-codices_sbe-0611_0001r_Layered.jpg", False) #path_dir, filename, with_color
    img = cv2.imread("tests/src/e-codices_sbe-0611_0001r_Layered.jpg")
    
    reduced_image = scaleImage(img, 6)
    
    showImage(reduced_image)
    cv2.waitKey(10) & 0xFF
    color = 0
    
    while(True):
        for i in range(100,200):
            for j in range(100,200):
                reduced_image[i,j,:] = color;
        
        if color == 0:
            color = 255
        else:
            color = 0
            
        showImage(reduced_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
    cv2.destroyAllWindows()
    