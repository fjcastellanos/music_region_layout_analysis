
from file_manager import FileManager
from MuretInterface import MuretInterface
import numpy as np
import cv2
import random
from skimage.filters import (threshold_sauvola)
import math
import skimage.transform as st
import json

#Generator of data augmentation images
class DataAugmentationGenerator:

    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        image_shape = image.shape
        h = image_shape[0]
        w = image_shape[1]
        if len(image_shape) == 3:
            c = image_shape[2]
            borderValue = (255,)*c
        else:
            borderValue = 255
        
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(
                            src=image.astype(np.float32), 
                            M=M, 
                            dsize=(w,h), 
                            borderValue=borderValue)
        return rotated, center


    @staticmethod
    def rotatePoint(origin, point, angle_grad):
        ox, oy = origin
        px, py = point

        angle_rad = math.radians(angle_grad)
        qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
        qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
        return int(qx), int(qy)
        
    @staticmethod
    def applyDataAugmentationToRegion(src_region, fixed_angle=None):

        if fixed_angle is None:
            angle = random.randint(-1,1)
        else:
            angle = fixed_angle

        rotated_src_region, center = DataAugmentationGenerator.rotate(src_region, angle)

        #FileManager.saveImageFullPath(rotated_src_region, "pruebas/region.png")
        return rotated_src_region, angle, center

    @staticmethod
    def generateNewImageByRandomSelection(img_orig, json_orig, list_json_pathfiles, vertical_region_resize):
        new_bbox_regions = {}
        new_img = np.copy(img_orig)
        #gt_img = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        
        dict_regions = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles = list_json_pathfiles, considered_classes = None)

        list_key_regions = [key_region for key_region in dict_regions for region in dict_regions[key_region] ]
        
        while (True):
            key_region = random.choice(list_key_regions)
            
            selected_patch = MuretInterface.selectRandomRegion(key_region, dict_regions)
            selected_patch_gray = cv2.cvtColor(selected_patch, cv2.COLOR_RGB2GRAY)
            thresh_sauvola = threshold_sauvola(image = selected_patch_gray, window_size=101, k=0.2)
            binary_sauvola = (selected_patch_gray > thresh_sauvola)*255

            break
    
        
        return new_img, new_bbox_regions
        

    @staticmethod
    def obtainBinaryImageBySauvola(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh_sauvola = threshold_sauvola(image = img_gray, window_size=101, k=0.2)
        binary_sauvola = (img_gray > thresh_sauvola)*255
        return binary_sauvola

    @staticmethod
    def extractCoordinatesFromBoundingBox(bbox_region):
        min_row_orig = bbox_region[0]
        max_row_orig = bbox_region[2]
        min_col_orig = bbox_region[1]
        max_col_orig = bbox_region[3]
        return min_row_orig, max_row_orig, min_col_orig, max_col_orig

    @staticmethod
    def createBoundingBox(min_row, max_row, min_col, max_col, new_img):
        min_row_new = max(0, min_row)
        max_row_new = min(new_img.shape[0]-1, max_row)
        min_col_new = max(0, min_col)
        max_col_new = min(new_img.shape[1]-1, max_col)
        new_bbox_region  = (min_row_new, min_col_new, max_row_new, max_col_new)
        return new_bbox_region

    @staticmethod
    def applyVerticalResize(selected_patch, bbox_region):

        min_row_orig, max_row_orig, min_col_orig, max_col_orig = DataAugmentationGenerator.extractCoordinatesFromBoundingBox(bbox_region)
        
        height_orig_bbox = max_row_orig - min_row_orig

        width_new_bbox = int((height_orig_bbox * selected_patch.shape[1]) / selected_patch.shape[0])

        selected_patch = cv2.resize(selected_patch,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)
        binary_sauvola = cv2.resize(selected_patch,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)

        min_row = bbox_region[0]
        max_row = min(bbox_region[0] + height_orig_bbox, new_img.shape[0])
        min_col = bbox_region[1]
        max_col = min(bbox_region[1] + width_orig_bbox, new_img.shape[1])

        return selected_patch, min_row, max_row, min_col, max_col
    
    @staticmethod
    def generateNewImage(img_orig, json_orig, list_json_pathfiles, vertical_region_resize):
        
        new_bbox_regions = {}
        new_img = np.copy(img_orig)
        #gt_img = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        
        dict_regions = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles = list_json_pathfiles, considered_classes = None)
        bboxes_orig = MuretInterface.readBoundingBoxes(list_json_pathfiles = list([json_orig]), considered_classes = None)

        #print (num_regions)
        #print (dict_regions)

        
        assert(len(bboxes_orig) == 1)
        idx_region=0
        for key in bboxes_orig:
            bboxes_regions = bboxes_orig[key]

            for key_region in bboxes_regions:
                print(key_region)

                for bbox_region in bboxes_regions[key_region]:
                    print (bbox_region)

                    selected_patch = MuretInterface.selectRandomRegion(key_region, dict_regions)
                    binary_sauvola = DataAugmentationGenerator.obtainBinaryImageBySauvola(selected_patch)

                    min_row_orig, max_row_orig, min_col_orig, max_col_orig = DataAugmentationGenerator.extractCoordinatesFromBoundingBox(bbox_region)
                    height_orig_bbox = max_row_orig - min_row_orig
                    width_orig_bbox = max_col_orig - min_col_orig
                    
                    min_row = bbox_region[0]
                    max_row = min(bbox_region[0] + binary_sauvola.shape[0], new_img.shape[0])
                    min_col = bbox_region[1]
                    max_col = min(bbox_region[1] + binary_sauvola.shape[1], new_img.shape[1])
                    height_new_bbox = max_row - min_row
                    width_new_bbox = max_col-min_col

                    max_row = min(max_row, max_row_orig)
                    max_col = min(max_col, max_col_orig)

                    if vertical_region_resize:
                        selected_patch, min_row, max_row, min_col, max_col = DataAugmentationGenerator.applyVerticalResize(selected_patch, bbox_region)
                        height_new_bbox = max_row - min_row
                        width_new_bbox = max_col-min_col
                    
                    selected_patch = selected_patch[0:height_new_bbox, 0:width_new_bbox]
                    selected_patch, angle, center = DataAugmentationGenerator.applyDataAugmentationToRegion(selected_patch)

                    binary_sauvola_patch = binary_sauvola[0:height_new_bbox, 0:width_new_bbox]
                    binary_sauvola_patch, _ = DataAugmentationGenerator.rotate(binary_sauvola_patch, angle)

                    binary_sauvola_patch = (binary_sauvola_patch > 128)*255

                    FileManager.saveImageFullPath(binary_sauvola_patch, "pruebas/sav_"+str(key_region) + str(idx_region) +".png")
                    FileManager.saveImageFullPath(selected_patch, "pruebas/" + str(key_region) + str(idx_region) +".png")
                    idx_region+=1
                    binary_sauvola_patch = binary_sauvola_patch.astype(np.uint8)

                    coords = np.where(binary_sauvola_patch == 0)

                    center_in_full_img = (center[0] + min_col, center[1]+min_row)

                    col_top_left, row_top_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, min_row), angle)
                    col_bot_left, row_bot_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, max_row), angle)

                    col_top_right, row_top_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, min_row), angle)
                    col_bot_right, row_bot_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, max_row), angle)

                    min_row_new = min(row_top_left, row_top_right)
                    min_col_new = min(col_top_left, col_bot_left)

                    max_row_new = max(row_bot_left, row_bot_right)
                    max_col_new = max(col_top_right, col_bot_right)

                    for idx in range(0,len(coords[0])):
                        coord_x = coords[0][idx]
                        coord_y = coords[1][idx]

                        if (min_row_new + coord_x) >= 0 and (min_row_new + coord_x) < new_img.shape[0] and (min_col_new + coord_y) >= 0 and (min_col_new + coord_y) < new_img.shape[1]: 
                            new_img[min_row_new + coord_x, min_col_new + coord_y] = selected_patch[coord_x, coord_y]

                    new_bbox_region = DataAugmentationGenerator.createBoundingBox(min_row_new, max_row_new, min_col_new, max_col_new, new_img)
                    
                    if key_region not in new_bbox_regions:
                        new_bbox_regions[key_region] = []

                    new_bbox_regions[key_region].append(new_bbox_region)

        return new_img, new_bbox_regions
    
    @staticmethod
    def generateNewImageOnPage(img_orig, json_orig, list_json_pathfiles, vertical_region_resize):
        
        new_bbox_regions = {}
        new_img = np.copy(img_orig)
        #gt_img = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        
        dict_regions = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles = list_json_pathfiles, considered_classes = None)
        bboxes_orig = MuretInterface.readBoundingBoxes(list_json_pathfiles = list([json_orig]), considered_classes = None)

        #print (num_regions)
        #print (dict_regions)

        
        assert(len(bboxes_orig) == 1)
        idx_region=0
        for key in bboxes_orig:
            bboxes_regions = bboxes_orig[key]

            for key_region in bboxes_regions:
                print(key_region)

                for bbox_region in bboxes_regions[key_region]:
                    print (bbox_region)

                    selected_patch = selected_img[bbox_region[0]:bbox_region[2], bbox_region[1]:bbox_region[3]]
                    binary_sauvola = DataAugmentationGenerator.obtainBinaryImageBySauvola(selected_patch)

                    min_row_orig, max_row_orig, min_col_orig, max_col_orig = DataAugmentationGenerator.extractCoordinatesFromBoundingBox(bbox_region)
                    height_orig_bbox = max_row_orig - min_row_orig
                    width_orig_bbox = max_col_orig - min_col_orig
                    
                    min_row = bbox_region[0]
                    max_row = min(bbox_region[0] + binary_sauvola.shape[0], new_img.shape[0])
                    min_col = bbox_region[1]
                    max_col = min(bbox_region[1] + binary_sauvola.shape[1], new_img.shape[1])
                    height_new_bbox = max_row - min_row
                    width_new_bbox = max_col-min_col

                    max_row = min(max_row, max_row_orig)
                    max_col = min(max_col, max_col_orig)

                    if vertical_region_resize:
                        selected_patch, min_row, max_row, min_col, max_col = DataAugmentationGenerator.applyVerticalResize(selected_patch, bbox_region)
                        height_new_bbox = max_row - min_row
                        width_new_bbox = max_col-min_col
                    
                    selected_patch = selected_patch[0:height_new_bbox, 0:width_new_bbox]
                    selected_patch, angle, center = DataAugmentationGenerator.applyDataAugmentationToRegion(selected_patch)

                    binary_sauvola_patch = binary_sauvola[0:height_new_bbox, 0:width_new_bbox]
                    binary_sauvola_patch, _ = DataAugmentationGenerator.rotate(binary_sauvola_patch, angle)

                    binary_sauvola_patch = (binary_sauvola_patch > 128)*255

                    FileManager.saveImageFullPath(binary_sauvola_patch, "pruebas/sav_"+str(key_region) + str(idx_region) +".png")
                    FileManager.saveImageFullPath(selected_patch, "pruebas/" + str(key_region) + str(idx_region) +".png")
                    idx_region+=1
                    binary_sauvola_patch = binary_sauvola_patch.astype(np.uint8)

                    coords = np.where(binary_sauvola_patch == 0)

                    center_in_full_img = (center[0] + min_col, center[1]+min_row)

                    col_top_left, row_top_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, min_row), angle)
                    col_bot_left, row_bot_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, max_row), angle)

                    col_top_right, row_top_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, min_row), angle)
                    col_bot_right, row_bot_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, max_row), angle)

                    min_row_new = min(row_top_left, row_top_right)
                    min_col_new = min(col_top_left, col_bot_left)

                    max_row_new = max(row_bot_left, row_bot_right)
                    max_col_new = max(col_top_right, col_bot_right)

                    for idx in range(0,len(coords[0])):
                        coord_x = coords[0][idx]
                        coord_y = coords[1][idx]

                        if (min_row_new + coord_x) >= 0 and (min_row_new + coord_x) < new_img.shape[0] and (min_col_new + coord_y) >= 0 and (min_col_new + coord_y) < new_img.shape[1]: 
                            new_img[min_row_new + coord_x, min_col_new + coord_y] = selected_patch[coord_x, coord_y]

                    new_bbox_region = DataAugmentationGenerator.createBoundingBox(min_row_new, max_row_new, min_col_new, max_col_new, new_img)
                    
                    if key_region not in new_bbox_regions:
                        new_bbox_regions[key_region] = []

                    new_bbox_regions[key_region].append(new_bbox_region)

        return new_img, new_bbox_regions

    @staticmethod
    def generateNewImageRandom(img_orig, json_orig, list_json_pathfiles, vertical_region_resize):
        
        new_bbox_regions = {}
        new_img = np.copy(img_orig)
        #gt_img = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        
        dict_regions = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles = list_json_pathfiles, considered_classes = None)
        bboxes_orig = MuretInterface.readBoundingBoxes(list_json_pathfiles = list([json_orig]), considered_classes = None)

        #print (num_regions)
        #print (dict_regions)

        
        assert(len(bboxes_orig) == 1)
        idx_region=0
        for key in bboxes_orig:
            bboxes_regions = bboxes_orig[key]

            for key_region in bboxes_regions:
                print(key_region)

                for bbox_region in bboxes_regions[key_region]:
                    print (bbox_region)

                    selected_patch = MuretInterface.selectRandomRegion(key_region, dict_regions)

                    selected_patch_gray = cv2.cvtColor(selected_patch, cv2.COLOR_RGB2GRAY)
                    thresh_sauvola = threshold_sauvola(image = selected_patch_gray, window_size=101, k=0.2)
                    binary_sauvola = (selected_patch_gray > thresh_sauvola)*255

                    min_row_orig = bbox_region[0]
                    max_row_orig = bbox_region[2]
                    min_col_orig = bbox_region[1]
                    max_col_orig = bbox_region[3]
                    height_orig_bbox = max_row_orig - min_row_orig
                    width_orig_bbox = max_col_orig - min_col_orig
                    
                    min_row = bbox_region[0]
                    max_row = min(bbox_region[0] + binary_sauvola.shape[0], new_img.shape[0])
                    min_col = bbox_region[1]
                    max_col = min(bbox_region[1] + binary_sauvola.shape[1], new_img.shape[1])
                    height_new_bbox = max_row - min_row
                    width_new_bbox = max_col-min_col

                    max_row = min(max_row, max_row_orig)
                    max_col = min(max_col, max_col_orig)

                    if vertical_region_resize:
                        width_new_bbox = int((height_orig_bbox * selected_patch.shape[1]) / selected_patch.shape[0])

                        selected_patch = cv2.resize(selected_patch,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)
                        binary_sauvola = cv2.resize(binary_sauvola,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)

                        min_row = bbox_region[0]
                        max_row = min(bbox_region[0] + height_orig_bbox, new_img.shape[0])
                        min_col = bbox_region[1]
                        max_col = min(bbox_region[1] + width_orig_bbox, new_img.shape[1])
                        height_new_bbox = max_row - min_row
                        width_new_bbox = max_col-min_col
                    
                    selected_patch = selected_patch[0:height_new_bbox, 0:width_new_bbox]
                    selected_patch, angle, center = DataAugmentationGenerator.applyDataAugmentationToRegion(selected_patch)

                    binary_sauvola_patch = binary_sauvola[0:height_new_bbox, 0:width_new_bbox]
                    binary_sauvola_patch, _ = DataAugmentationGenerator.rotate(binary_sauvola_patch, angle)

                    binary_sauvola_patch = (binary_sauvola_patch > 128)*255

                    FileManager.saveImageFullPath(binary_sauvola_patch, "pruebas/sav_"+str(key_region) + str(idx_region) +".png")
                    FileManager.saveImageFullPath(selected_patch, "pruebas/" + str(key_region) + str(idx_region) +".png")
                    idx_region+=1
                    binary_sauvola_patch = binary_sauvola_patch.astype(np.uint8)

                    coords = np.where(binary_sauvola_patch == 0)

                    center_in_full_img = (center[0] + min_col, center[1]+min_row)

                    col_top_left, row_top_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, min_row), angle)
                    col_bot_left, row_bot_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, max_row), angle)

                    col_top_right, row_top_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, min_row), angle)
                    col_bot_right, row_bot_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, max_row), angle)

                    min_row_new = min(row_top_left, row_top_right)
                    min_col_new = min(col_top_left, col_bot_left)

                    max_row_new = max(row_bot_left, row_bot_right)
                    max_col_new = max(col_top_right, col_bot_right)

                    for idx in range(0,len(coords[0])):
                        coord_x = coords[0][idx]
                        coord_y = coords[1][idx]

                        if (min_row_new + coord_x) >= 0 and (min_row_new + coord_x) < new_img.shape[0] and (min_col_new + coord_y) >= 0 and (min_col_new + coord_y) < new_img.shape[1]: 
                            new_img[min_row_new + coord_x, min_col_new + coord_y] = selected_patch[coord_x, coord_y]


                    min_row_new = max(0, min_row_new)
                    max_row_new = min(new_img.shape[0]-1, max_row_new)
                    min_col_new = max(0, min_col_new)
                    max_col_new = min(new_img.shape[1]-1, max_col_new)

                    #gt_img[min_row_new:max_row_new, min_col_new:max_col_new] = 1
                    #FileManager.saveImageFullPath(gt_img*255, "pruebas/imggt.png")
                    #FileManager.saveImageFullPath(new_img, "pruebas/img.png")

                    new_bbox_region  = (min_row_new, min_col_new, max_row_new, max_col_new)

                    if key_region not in new_bbox_regions:
                        new_bbox_regions[key_region] = []

                    new_bbox_regions[key_region].append(new_bbox_region)

        return new_img, new_bbox_regions

    @staticmethod
    def countNumberItemsList(mylist):
        num_elements = 0

        for item in mylist:
            if type(item) is list:
                num_elements += DataAugmentationGenerator.countNumberItemsList(item)
            else:
                num_elements += 1
        return num_elements

    @staticmethod
    def countNumberBBoxes(dict_regions, key):
        number_elements = 0
        for item in dict_regions[key]:
            if (type(dict_regions[key][item]) is list):
                number_elements += DataAugmentationGenerator.countNumberItemsList(dict_regions[key][item])
            else:
                number_elements += 1

        return number_elements

    @staticmethod
    def generateNewImageRandomAuto(img_orig, json_orig, list_json_pathfiles, vertical_region_resize, uniform_rotate):
        
        new_bbox_regions = {}
        new_img = np.copy(img_orig)
        gt_img = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        
        dict_regions = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles = list_json_pathfiles, considered_classes = None)
        bboxes_orig = MuretInterface.readBoundingBoxes(list_json_pathfiles = list([json_orig]), considered_classes = None)

        number_staves = DataAugmentationGenerator.countNumberBBoxes(dict_regions, "staff")
        number_empty_staves = DataAugmentationGenerator.countNumberBBoxes(dict_regions, "empty_staff")
        number_lyrics = DataAugmentationGenerator.countNumberBBoxes(dict_regions, "lyrics")

        print("Number of pages: " + str(len(list_json_pathfiles)))
        print("Lyrics: " + str(number_lyrics))
        print("Staves: " + str(number_staves))
        print("Empty staves: " + str(number_empty_staves))
        print("Total staves: " + str(number_staves + number_empty_staves))

        if (uniform_rotate is True):
            uniform_angle = random.randint(-2,2)
        else:
            uniform_angle = None

        #print (num_regions)
        #print (dict_regions)

        overlapped = 0
        assert(len(bboxes_orig) == 1)
        idx_region=0
        min_row_last = 100
        min_col_last = 100
        max_row_last = 0
        max_col_last = 0

        idx_region_staff = 1
        dict_keys_value = {}
        for key in bboxes_orig:
            bboxes_regions = bboxes_orig[key]

            idx_region_key = 1
            for key_region in bboxes_regions:
                print(key_region)

                if key_region not in dict_keys_value:
                    if key_region == "staff":
                        idx_region_staff = idx_region_key

                    if "staff" in key_region:
                        dict_keys_value[key_region] = idx_region_staff
                    else:
                        dict_keys_value[key_region] = idx_region_key
                    
                    idx_region_key += 1
                    

                for bbox_region in bboxes_regions[key_region]:
                    print (bbox_region)

                    selected_patch = MuretInterface.selectRandomRegion(key_region, dict_regions)

                    selected_patch_gray = cv2.cvtColor(selected_patch, cv2.COLOR_RGB2GRAY)
                    thresh_sauvola = threshold_sauvola(image = selected_patch_gray, window_size=101, k=0.2)
                    binary_sauvola = (selected_patch_gray > thresh_sauvola)*255

                    min_row_orig = bbox_region[0]
                    max_row_orig = bbox_region[2]
                    min_col_orig = bbox_region[1]
                    max_col_orig = bbox_region[3]
                    height_orig_bbox = max_row_orig - min_row_orig
                    width_orig_bbox = max_col_orig - min_col_orig
                    
                    min_row = bbox_region[0]
                    max_row = min(bbox_region[0] + binary_sauvola.shape[0], new_img.shape[0])
                    min_col = bbox_region[1]
                    max_col = min(bbox_region[1] + binary_sauvola.shape[1], new_img.shape[1])
                    height_new_bbox = max_row - min_row
                    width_new_bbox = max_col-min_col

                    max_row = min(max_row, max_row_orig)
                    max_col = min(max_col, max_col_orig)

                    if vertical_region_resize:
                        width_new_bbox = int((height_orig_bbox * selected_patch.shape[1]) / selected_patch.shape[0])

                        selected_patch = cv2.resize(selected_patch,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)
                        binary_sauvola = cv2.resize(binary_sauvola,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)

                        min_row = bbox_region[0]
                        max_row = min(bbox_region[0] + height_orig_bbox, new_img.shape[0])
                        min_col = bbox_region[1]
                        max_col = min(bbox_region[1] + width_orig_bbox, new_img.shape[1])
                        height_new_bbox = max_row - min_row
                        width_new_bbox = max_col-min_col
                    
                    selected_patch = selected_patch[0:height_new_bbox, 0:width_new_bbox]
                    selected_patch, angle, center = DataAugmentationGenerator.applyDataAugmentationToRegion(selected_patch, uniform_angle)

                    binary_sauvola_patch = binary_sauvola[0:height_new_bbox, 0:width_new_bbox]
                    binary_sauvola_patch, _ = DataAugmentationGenerator.rotate(binary_sauvola_patch, angle)

                    binary_sauvola_patch = (binary_sauvola_patch > 128)*255

                    #FileManager.saveImageFullPath(binary_sauvola_patch, "pruebas/sav_"+str(key_region) + str(idx_region) +".png")
                    #FileManager.saveImageFullPath(selected_patch, "pruebas/" + str(key_region) + str(idx_region) +".png")
                    idx_region+=1
                    binary_sauvola_patch = binary_sauvola_patch.astype(np.uint8)

                    coords = np.where(binary_sauvola_patch == 0)

                    center_in_full_img = (center[0] + min_col, center[1]+min_row)

                    min_col_new = center_in_full_img[0] - binary_sauvola_patch.shape[1]//2
                    min_row_new = center_in_full_img[1] - binary_sauvola_patch.shape[0]//2
                    max_col_new = center_in_full_img[0] + binary_sauvola_patch.shape[1]//2
                    max_row_new = center_in_full_img[1] + binary_sauvola_patch.shape[0]//2

                    #col_top_left, row_top_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, min_row), angle)
                    #col_bot_left, row_bot_left = DataAugmentationGenerator.rotatePoint(center_in_full_img, (min_col, max_row), angle)

                    #col_top_right, row_top_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, min_row), angle)
                    #col_bot_right, row_bot_right = DataAugmentationGenerator.rotatePoint(center_in_full_img, (max_col, max_row), angle)

                    #min_row_new = min(row_top_left, row_top_right)
                    #min_col_new = min(col_top_left, col_bot_left)

                    #max_row_new = max(row_bot_left, row_bot_right)
                    #max_col_new = max(col_top_right, col_bot_right)

                    min_row_new = max(0, min_row_new)
                    max_row_new = min(new_img.shape[0]-1, max_row_new)
                    min_col_new = max(0, min_col_new)
                    max_col_new = min(new_img.shape[1]-1, max_col_new)

                    

                    labels = np.unique(gt_img[min_row_new:max_row_new, min_col_new:max_col_new])
                    
                    if (dict_keys_value[key_region] in labels): #Overlapping with other bounding box
                        print ("Overlapping!")
                        if (overlapped > 10):
                            break
                        overlapped+=1
                        continue

                    overlapped = 0
                    
                    for idx in range(0,len(coords[0])):
                        coord_x = coords[0][idx]
                        coord_y = coords[1][idx]

                        if (min_row_new + coord_x) >= 0 and (min_row_new + coord_x) < new_img.shape[0] and (min_col_new + coord_y) >= 0 and (min_col_new + coord_y) < new_img.shape[1]: 
                            new_img[min_row_new + coord_x, min_col_new + coord_y] = selected_patch[coord_x, coord_y]


                    gt_img[min_row_new:max_row_new, min_col_new:max_col_new] = dict_keys_value[key_region]

                    #FileManager.saveImageFullPath((gt_img>0)*255, "pruebas/imggt.png")
                    #FileManager.saveImageFullPath(new_img, "pruebas/img.png")

                    new_bbox_region  = (min_row_new, min_col_new, max_row_new, max_col_new)

                    if key_region not in new_bbox_regions:
                        new_bbox_regions[key_region] = []

                    new_bbox_regions[key_region].append(new_bbox_region)

        return new_img, new_bbox_regions

    @staticmethod
    def generateNewImageFromListByRandomSelection(json_dataset, number_new_images, vertical_region_resize = False, json_dirpath_out=None, reduction_factor = 0.3):
        if json_dirpath_out is None:
            json_dirpath_out = json_dataset.replace("datasets/", "datasets/daug/")

        src_dirpath_out = json_dirpath_out.replace("/JSON/", "/SRC/")

        list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)
        for idx_new_image in range(number_new_images):

            with_regions = False
            while (with_regions == False):
                json_pathfile = random.choice(list_json_pathfiles)

                img_pathfile = json_pathfile.replace(".json", "").replace("JSON", "SRC")

                print("ID: " + str(idx_new_image) + "-blur from: " + str(img_pathfile))
                img = FileManager.loadImage(img_pathfile, True)
                blur_img = MuretInterface.applyBlurring(img=img, factor = 2)
                new_img, new_bbox_regions = DataAugmentationGenerator.generateNewImageByRandomSelection(blur_img, json_pathfile, list_json_pathfiles, vertical_region_resize)
                
                if len(new_bbox_regions) > 0:
                    with_regions = True

            src_filepath_out = src_dirpath_out + str(idx_new_image) + ".png"
            gt_filepath_out = src_filepath_out.replace("/SRC/", "/GT/")
            json_filepath_out = src_filepath_out.replace("/SRC/", "/JSON/").replace(".png", ".dict")
            
            FileManager.saveImageFullPath(new_img, src_filepath_out)
            print("Saved data augmentation image in " + str(src_filepath_out) + " (from " + str(img_pathfile) + ")")

            gt_imgs = MuretInterface.generateGTImages(new_bbox_regions, (new_img.shape[0], new_img.shape[1]), reduction_factor)
            idx_gt = 0
            for key_region in new_bbox_regions:
                FileManager.saveImageFullPath(gt_imgs[idx_gt]*255, gt_filepath_out + "_" + str(key_region) + ".png")
                idx_gt+=1

            #json_muret = json.dumps(new_bbox_regions, indent = 4)
            
            src_filename = FileManager.nameOfFileWithExtension(src_filepath_out)

            json_muret = MuretInterface.generateJSONMURET(new_bbox_regions, json_pathfile, src_filename, new_img.shape)
            FileManager.saveString(str(json_muret), json_filepath_out, True)


    @staticmethod
    def generateNewImageFromListByReplacingBoundingBoxes(daug_type, json_dataset, number_new_images, vertical_region_resize = False, json_dirpath_out=None, reduction_factor = 0.3):

        if json_dirpath_out is None:
            json_dirpath_out = json_dataset.replace("datasets/", "datasets/daug/" + str(daug_type) + "/")

        src_dirpath_out = json_dirpath_out.replace("/JSON/", "/SRC/")

        list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)

        for idx_new_image in range(number_new_images):
            with_regions = False
            while (with_regions == False):
                json_pathfile = random.choice(list_json_pathfiles)

                img_pathfile = json_pathfile.replace(".json", "").replace("JSON", "SRC")

                print("ID: " + str(idx_new_image) + "-blur from: " + str(img_pathfile))
                img = FileManager.loadImage(img_pathfile, True)
                blur_img = MuretInterface.applyBlurring(img=img, factor = 2)
                new_img, new_bbox_regions = DataAugmentationGenerator.generateNewImage(blur_img, json_pathfile, list_json_pathfiles, vertical_region_resize)

                if len(new_bbox_regions) > 0:
                    with_regions = True

            src_filepath_out = src_dirpath_out + str(idx_new_image) + ".png"
            gt_filepath_out = src_filepath_out.replace("/SRC/", "/GT/")
            json_filepath_out = src_filepath_out.replace("/SRC/", "/JSON/").replace(".png", ".dict")
            
            FileManager.saveImageFullPath(new_img, src_filepath_out)
            print("Saved data augmentation image in " + str(src_filepath_out) + " (from " + str(img_pathfile) + ")")

            gt_imgs = MuretInterface.generateGTImages(new_bbox_regions, (new_img.shape[0], new_img.shape[1]), reduction_factor)
            idx_gt = 0
            for key_region in new_bbox_regions:
                FileManager.saveImageFullPath(gt_imgs[idx_gt]*255, gt_filepath_out + "_" + str(key_region) + ".png")
                idx_gt+=1

            #json_muret = json.dumps(new_bbox_regions, indent = 4)
            
            src_filename = FileManager.nameOfFileWithExtension(src_filepath_out)

            json_muret = MuretInterface.generateJSONMURET(new_bbox_regions, json_pathfile, src_filename, new_img.shape)
            FileManager.saveString(str(json_muret), json_filepath_out, True)


    @staticmethod
    def generateNewImageFromListByReplacingBoundingBoxesOnPage(daug_type, json_dataset, parent_dir_str, number_new_images, uniform_rotate = False, vertical_region_resize = False, json_dirpath_out=None, reduction_factor = 0.3):

        if uniform_rotate is True:
            daug_type = daug_type + "-uniform-rotate"
        
        if json_dataset is not None and type(json_dataset) is list:
            #It is a list of files
            list_json_pathfiles = json_dataset
            json_dataset = parent_dir_str.replace("Folds", "")+ "/"
        else:
            #It is a directory
            list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)

        if num_pages > 0:
            assert(num_pages < len(list_json_pathfiles))
            list_json_pathfiles = list_json_pathfiles[0:num_pages]

        
        json_files = json.dumps(str(list_json_pathfiles), indent = 4)

        if json_dirpath_out is None:
            json_dirpath_out = json_dataset.replace("datasets/", "datasets/daug/" + str(daug_type) + "/" + str(num_pages)+"_pages" + "/"+ str(number_new_images) + "_n" + "/Folds/fold" + str(fold) +  "/")
            src_dirpath_out = json_dirpath_out.replace("/JSON/", "/SRC/")
            
        FileManager.saveString(str(json_files), json_dirpath_out.replace("JSON/", "LISTFILES/") + "files_considered.json", True)

        for idx_new_image in range(number_new_images):
            with_regions = False
            while (with_regions == False):
                json_pathfile = random.choice(list_json_pathfiles)

                img_pathfile = json_pathfile.replace(".json", "").replace("JSON", "SRC")

                print("ID: " + str(idx_new_image) + "-blur from: " + str(img_pathfile))
                img = FileManager.loadImage(img_pathfile, True)
                blur_img = MuretInterface.applyBlurring(img=img, factor = 2)
                new_img, new_bbox_regions = DataAugmentationGenerator.generateNewImage(blur_img, json_pathfile, list_json_pathfiles, vertical_region_resize)

                if len(new_bbox_regions) > 0:
                    with_regions = True

            src_filepath_out = src_dirpath_out + str(idx_new_image) + ".png"
            gt_filepath_out = src_filepath_out.replace("/SRC/", "/GT/")
            json_filepath_out = src_filepath_out.replace("/SRC/", "/JSON/").replace(".png", ".dict")
            
            FileManager.saveImageFullPath(new_img, src_filepath_out)
            print("Saved data augmentation image in " + str(src_filepath_out) + " (from " + str(img_pathfile) + ")")

            gt_imgs = MuretInterface.generateGTImages(new_bbox_regions, (new_img.shape[0], new_img.shape[1]), reduction_factor)
            idx_gt = 0
            for key_region in new_bbox_regions:
                FileManager.saveImageFullPath(gt_imgs[idx_gt]*255, gt_filepath_out + "_" + str(key_region) + ".png")
                idx_gt+=1

            #json_muret = json.dumps(new_bbox_regions, indent = 4)
            
            src_filename = FileManager.nameOfFileWithExtension(src_filepath_out)

            json_muret = MuretInterface.generateJSONMURET(new_bbox_regions, json_pathfile, src_filename, new_img.shape)
            FileManager.saveString(str(json_muret), json_filepath_out, True)

    @staticmethod
    def generateNewImageFromListByBoundingBoxesRandomSelection(daug_type, num_pages, json_dataset, parent_dir_str, fold, number_new_images, uniform_rotate = False, vertical_region_resize = False, json_dirpath_out=None, reduction_factor = 0.3):

        if uniform_rotate is True:
            daug_type = daug_type + "-uniform-rotate"
        
        if json_dataset is not None and type(json_dataset) is list:
            #It is a list of files
            list_json_pathfiles = json_dataset
            json_dataset = parent_dir_str.replace("Folds", "")+ "/"
        else:
            #It is a directory
            list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)

        if num_pages > 0:
            assert(num_pages < len(list_json_pathfiles))
            list_json_pathfiles = list_json_pathfiles[0:num_pages]

        
        json_files = json.dumps(str(list_json_pathfiles), indent = 4)

        if json_dirpath_out is None:
            json_dirpath_out = json_dataset.replace("datasets/", "datasets/daug/" + str(daug_type) + "/" + str(num_pages)+"_pages" + "/"+ str(number_new_images) + "_n" + "/Folds/fold" + str(fold) +  "/")
            src_dirpath_out = json_dirpath_out.replace("/JSON/", "/SRC/")
            
        FileManager.saveString(str(json_files), json_dirpath_out.replace("JSON/", "LISTFILES/") + "files_considered.json", True)

        for idx_new_image in range(number_new_images):
            with_regions = False
            while (with_regions == False):
                json_pathfile = random.choice(list_json_pathfiles)

                img_pathfile = json_pathfile.replace(".json", "").replace("JSON", "SRC")

                print("ID: " + str(idx_new_image) + "-blur from: " + str(img_pathfile))
                img = FileManager.loadImage(img_pathfile, True)
                blur_img = MuretInterface.applyBlurring(img=img, factor = 2)
                new_img, new_bbox_regions = DataAugmentationGenerator.generateNewImageRandom(blur_img, json_pathfile, list_json_pathfiles, vertical_region_resize)

                if len(new_bbox_regions) > 0:
                    with_regions = True

            src_filepath_out = src_dirpath_out + str(idx_new_image) + ".png"
            gt_filepath_out = src_filepath_out.replace("/SRC/", "/GT/")
            json_filepath_out = src_filepath_out.replace("/SRC/", "/JSON/").replace(".png", ".dict")
            
            FileManager.saveImageFullPath(new_img, src_filepath_out)
            print("Saved data augmentation image in " + str(src_filepath_out) + " (from " + str(img_pathfile) + ")")

            gt_imgs = MuretInterface.generateGTImages(new_bbox_regions, (new_img.shape[0], new_img.shape[1]), reduction_factor)
            idx_gt = 0
            for key_region in new_bbox_regions:
                FileManager.saveImageFullPath(gt_imgs[idx_gt]*255, gt_filepath_out + "_" + str(key_region) + ".png")
                idx_gt+=1

            #json_muret = json.dumps(new_bbox_regions, indent = 4)
            
            src_filename = FileManager.nameOfFileWithExtension(src_filepath_out)

            json_muret = MuretInterface.generateJSONMURET(new_bbox_regions, json_pathfile, src_filename, new_img.shape)
            FileManager.saveString(str(json_muret), json_filepath_out, True)
        

    @staticmethod
    def generateNewImageFromListByBoundingBoxesRandomSelectionAuto(daug_type, num_pages, json_dataset, parent_dir_str, fold, number_new_images, uniform_rotate = False, vertical_region_resize = False, json_dirpath_out=None, reduction_factor = 0.3):

        if uniform_rotate is True:
            daug_type = daug_type + "-uniform-rotate"
        
        if json_dataset is not None and type(json_dataset) is list:
            #It is a list of files
            list_json_pathfiles = json_dataset
            json_dataset = parent_dir_str.replace("Folds", "") + "/"
        else:
            #It is a directory
            list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)

        if num_pages > 0:
            assert(num_pages < len(list_json_pathfiles))
            list_json_pathfiles = list_json_pathfiles[0:num_pages]

        
        json_files = json.dumps(str(list_json_pathfiles), indent = 4)

        if json_dirpath_out is None:
            json_dirpath_out = json_dataset.replace("datasets/", "datasets/daug/" + str(daug_type) + "/" + str(num_pages)+"_pages" + "/"+ str(number_new_images) + "_n" + "/Folds/fold" + str(fold) +  "/")
            src_dirpath_out = json_dirpath_out.replace("/JSON/", "/SRC/")
            
        #FileManager.saveString(str(json_files), json_dirpath_out.replace("JSON/", "LISTFILES/") + "files_considered.json", True)


        for idx_new_image in range(number_new_images):
            with_regions = False
            while (with_regions == False):    
                json_pathfile = random.choice(list_json_pathfiles)

                img_pathfile = json_pathfile.replace(".json", "").replace("JSON", "SRC")

                print("ID: " + str(idx_new_image) + "-blur from: " + str(img_pathfile))
                img = FileManager.loadImage(img_pathfile, True)
                blur_img = MuretInterface.applyBlurring(img=img, factor = 2)
                new_img, new_bbox_regions = DataAugmentationGenerator.generateNewImageRandomAuto(blur_img, json_pathfile, list_json_pathfiles, vertical_region_resize, uniform_rotate)

                if len(new_bbox_regions) > 0:
                    with_regions = True

            src_filepath_out = src_dirpath_out + str(idx_new_image) + ".png"
            gt_filepath_out = src_filepath_out.replace("/SRC/", "/GT/")
            json_filepath_out = src_filepath_out.replace("/SRC/", "/JSON/").replace(".png", ".dict")
            
            FileManager.saveImageFullPath(new_img, src_filepath_out)
            print("Saved data augmentation image in " + str(src_filepath_out) + " (from " + str(img_pathfile) + ")")

            gt_imgs = MuretInterface.generateGTImages(new_bbox_regions, (new_img.shape[0], new_img.shape[1]), reduction_factor)
            idx_gt = 0
            for key_region in new_bbox_regions:
                FileManager.saveImageFullPath(gt_imgs[idx_gt]*255, gt_filepath_out + "_" + str(key_region) + ".png")
                idx_gt+=1

            #json_muret = json.dumps(new_bbox_regions, indent = 4)
            
            src_filename = FileManager.nameOfFileWithExtension(src_filepath_out)

            json_muret = MuretInterface.generateJSONMURET(new_bbox_regions, json_pathfile, src_filename, new_img.shape)
            FileManager.saveString(str(json_muret), json_filepath_out, True)

if __name__ == "__main__":
    random.seed(42)
    DataAugmentationGenerator.generateNewImageFromListByRandomSelection("datasets/Folds/Fold0/Train/JSON/Mus-Tradicional/c-combined/c5/", 100, False)

