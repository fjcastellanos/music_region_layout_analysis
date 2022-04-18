
from DataAugmentationGenerator import DataAugmentationGenerator
from file_manager import FileManager
import cv2
import numpy as np
import random
import argparse
import skimage.transform as st
from shutil import copyfile



def menu():
    parser = argparse.ArgumentParser(description='DA SAE')
    parser.add_argument('-type',   default='random-auto', type=str,     choices=['replacing', 'on-page', 'random-auto', 'random'],  help='Training type')
    parser.add_argument('-n',   default=100, type=int,  help='Number of images to be generated')
    parser.add_argument('-pages',   default=0, type=int,  help='Number of real pages to be considered')
    parser.add_argument('-jsons', type=str, default=None,  help='Path to the GT dataset directory (json)')
    parser.add_argument('-txt_train', type=str,  default=None, help='Path to the txt file with the paths of the images.')
    parser.add_argument('--vrs',   action='store_true', help='Active the vertical resize of regions')
    parser.add_argument('--folds',   action='store_true', help='Generate folds')
    parser.add_argument('--uniform_rotate',   action='store_true', help='Uniform the rotation for each page')
    parser.add_argument('-seed',   default=42, type=int,  help='Seed')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


def generatePartitions(list_pathfiles, folds=5):
    dict_partitions = {}

    for idx_partition in range(folds):
        dict_partitions[idx_partition] = []

    idx_partition = 0
    for pathfile in list_pathfiles:
        dict_partitions[idx_partition].append(pathfile)
        idx_partition = (idx_partition+1)%folds

    return dict_partitions


def replace(list_string_items, string1, string2):

    list_replaced = []
    for string_item in list_string_items:

        list_replaced.append(string_item.replace(string1, string2))

    return list_replaced

def getPathfilesFold_out(pathfile_json_orig, fold, subfolder):
    return pathfile_json_orig.replace("datasets/", "datasets/Folds/Fold" + str(fold) + "/" + subfolder + "/")

def copyFilesToFolds(json_test_files, idx_partition, subfolder):
    for json_test_file in json_test_files:
        src_test_file = json_test_file.replace("/JSON/", "/SRC/").replace(".json", "")
        json_test_file_out = getPathfilesFold_out(json_test_file, idx_partition, subfolder)
        src_test_file_out = getPathfilesFold_out(src_test_file, idx_partition, subfolder)

        FileManager.copyFile(src_test_file, src_test_file_out)
        FileManager.copyFile(json_test_file, json_test_file_out)

def showStadistics(list_json_pathfiles):
    from MuretInterface import MuretInterface
    r = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles, considered_classes = ["staff", "empty-staff"])
    bboxes = [item2 for key_class in r for key_page in r[key_class] for item in r[key_class][key_page] for item2 in item]
    print("Bboxes: " + str(len(bboxes)))
    files = []
    files = [f for key_class in r for f in r[key_class] if f not in files]

    w = 0
    h = 0
    for f in files:
        f2 = f.replace("/JSON/", "/SRC/").replace(".json", "")
        im = FileManager.loadImage(f2, False)
        h += im.shape[0]
        w += im.shape[1]

    h /= len(files)
    w /= len(files)
    print("Files: " + str(len(files)))
    print ("Avg. Size: " + str(int(h)) + "x" + str(int(w)) + " px")

    f0_json = list(r["staff"].keys())[0]
    f0_src = f0_json.replace("/JSON/", "/SRC/").replace(".json", "")
    im = FileManager.loadImage(f0_src, True)
    gt = np.zeros((im.shape[0], im.shape[1]))

    for key_class in r:
        if f0_json in r[key_class]:
            for page in r[key_class][f0_json]:
                for bbox in page:
                    gt[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0.5

    for key_class in r:
        if f0_json in r[key_class]:
            for page in r[key_class][f0_json]:
                for bbox in page:
                    gt[bbox[0]:bbox[2], bbox[1]:bbox[1]+10] = 1
                    gt[bbox[0]:bbox[2], bbox[3]-10:bbox[3]] = 1
                    gt[bbox[0]:bbox[0]+10, bbox[1]:bbox[3]] = 1
                    gt[bbox[2]-10:bbox[2], bbox[1]:bbox[3]] = 1
    
    FileManager.saveImageFullPath(gt*255, "prueba_gt.png")
    FileManager.saveImageFullPath(im, "prueba_src.png")


def generateFolds(json_dataset, test_ratio=0.2, folds=5):

    list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)
    showStadistics(list_json_pathfiles)
    
    random.shuffle(list_json_pathfiles)

    dict_partitions = generatePartitions(list_json_pathfiles, folds)
    
    for test_partition, list_pathfiles_partition in dict_partitions.items():
        num_files_partition = len(list_pathfiles_partition)
        
        val_partition = (test_partition + 1) % folds

        train_partitions = [fold for fold in range(folds) if fold != test_partition and fold != val_partition]

        print('*'*80)
        print("Test partition: " + str(test_partition) + " -> " + str(num_files_partition) + " elements")
        print("Val partitions: " + str(val_partition))
        print("Train partitions: " + str(train_partitions))
        print('*'*80)

        json_train_files = [pathfile for train_partition in train_partitions for pathfile in dict_partitions[train_partition]]
        json_test_files = [pathfile for pathfile in dict_partitions[test_partition]]
        json_val_files = [pathfile for pathfile in dict_partitions[val_partition]]

        str_json_train_files = '\n'.join(json_train_files)
        str_json_test_files = '\n'.join(json_test_files)
        str_json_val_files = '\n'.join(json_val_files)

        #FileManager.saveString(str_json_train_files, True)

        pathfile_json_files = json_test_files[0].replace("datasets/JSON/", "datasets/JSON/Folds/")

        pathfile_json_files_test = json_dataset.replace("JSON/", "JSON/Folds/") + "fold" + str(test_partition) + "/test" + ".txt"
        pathfile_json_files_val = json_dataset.replace("JSON/", "JSON/Folds/") + "fold" + str(test_partition) + "/val" + ".txt"
        pathfile_json_files_train = json_dataset.replace("JSON/", "JSON/Folds/") + "fold" + str(test_partition) + "/train" + ".txt"
        

        FileManager.saveString(str_json_test_files, pathfile_json_files_test, True)
        FileManager.saveString(str_json_val_files, pathfile_json_files_val, True)
        FileManager.saveString(str_json_train_files, pathfile_json_files_train, True)

        #copyFilesToFolds(json_test_files, test_partition, "Test")
        #copyFilesToFolds(json_train_files, test_partition, "Train")
        




if __name__ == "__main__":
    config = menu()

    if config.folds:
        random.seed(config.seed)
        generateFolds(json_dataset=config.jsons, folds=5)
    
    else:
        random.seed(config.seed)

        if config.jsons is None:
            assert(config.txt_train is not None)
            content_txt_train_path_files = FileManager.readStringFile(config.txt_train)

            jsons = list(content_txt_train_path_files.split("\n"))

            from pathlib import Path
            parent_dir = Path(config.txt_train).parent
            fold = int((parent_dir.stem).replace("fold", ""))
            
            parent_dir_str = str(parent_dir.parent)

        else:
            jsons = config.jsons
            fold = None
            parent_dir_str = None

        if config.type == "replacing":
            DataAugmentationGenerator.generateNewImageFromListByReplacingBoundingBoxes(config.type, jsons, config.n, config.vrs)
        elif config.type == "random":
            DataAugmentationGenerator.generateNewImageFromListByBoundingBoxesRandomSelection(config.type, jsons, parent_dir_str, config.n, config.vrs)
        elif config.type == "on-page":
            DataAugmentationGenerator.generateNewImageFromListByReplacingBoundingBoxesOnPage(config.type, config.pages, jsons, parent_dir_str, fold, config.n, config.uniform_rotate, config.vrs)
        elif config.type == "random-auto":
            DataAugmentationGenerator.generateNewImageFromListByBoundingBoxesRandomSelectionAuto(config.type, config.pages, jsons, parent_dir_str, fold, config.n, config.uniform_rotate, config.vrs)
        else:
            assert(False)
