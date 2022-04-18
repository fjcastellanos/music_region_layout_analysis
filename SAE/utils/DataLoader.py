

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from random import randint
import random as rd
import json
from keras.utils import to_categorical

try:
    from utils.GTJSONReaderMuret import GTJSONReaderMuret
    from utils.file_manager import FileManager
    from utils.Document import Document
    from utils.Databases import getPathdirDatabase, getPathdirDatabaseSRCAndJSON_MURET
    from utils.ImageProcessing import *
    from utils.image_manager import *
    from utils.CustomJson import CustomJson

except:
    from GTJSONReaderMuret import GTJSONReaderMuret
    from file_manager import FileManager
    from Document import Document
    from Databases import getPathdirDatabase, getPathdirDatabaseSRCAndJSON_MURET
    from ImageProcessing import *
    from image_manager import *
    from CustomJson import CustomJson



def get3DShapeImage(image):
    im_shape = image.shape

    if len(im_shape) == 2:
        return (im_shape[0], im_shape[1], 1)
    else:
        return im_shape


def getRegionsFromJSON(json_path_file, considered_classes):

    list_regions = []
    with open(json_path_file) as json_file:
        data = json.load(json_file)
        if "pages" in data:

            for page in data["pages"]:
                if "regions" in page:
                    for region in page["regions"]:
                        if region["type"] in considered_classes:
                            list_regions.append(region)

    return list_regions


def generate_GroundTruth(list_regions, shape, reduction_GT):

    gt = np.zeros((shape[0], shape[1]))

    for region in list_regions:
        
        bbox = region["bounding_box"]
        fromX = bbox['fromX']
        toX = bbox['toX']
        fromY = bbox['fromY']
        toY = bbox['toY']

        vertical_height = toY - fromY
        reduction_px = (int(vertical_height * reduction_GT) )
        
        fromY_new = fromY + reduction_px
        toY_new = toY - reduction_px
        
        gt[fromY_new:toY_new, fromX:toX] = 1

    return gt

def get_image_with_gt(img_path_file, json_path_file, with_color, considered_classes, reduction_GT):

    gr = (255-FileManager.loadImage(img_path_file, with_color)) / 255.

    list_regions = getRegionsFromJSON(json_path_file, considered_classes)
    
    gt = generate_GroundTruth(list_regions, gr.shape, reduction_GT)
    gt = to_categorical(gt, num_classes=2)

    return gr, gt


def resizeImage(img, height, width, interpolation = cv2.INTER_LINEAR):
    img2 = img.copy()
    return cv2.resize(img2,(width,height), interpolation=interpolation)


def createGeneratorResizingImagesTest(list_json_files, patch_height, patch_width, batch_size, with_color, considered_classes, reduction_GT):
    
    gr_chunks = []
    gt_chunks = []

    count = 0

    for json_file_selected in list_json_files:

        src_file_selected = json_file_selected.replace("JSON/", "SRC/").replace(".json", "")

        gr, gt = get_image_with_gt(src_file_selected, json_file_selected, with_color, considered_classes, reduction_GT)

        gr = resizeImage(gr, patch_height, patch_width)
        gt = resizeImage(gt, patch_height, patch_width)

        gr_chunks.append(gr)
        gt_chunks.append(gt)

        gr_chunks_arr = np.array(gr_chunks)
        gt_chunks_arr = np.array(gt_chunks)
        # convert gr_chunks and gt_chunks to the numpy arrays that are yield below

        yield gr_chunks_arr  # convert into npy before yielding
        gr_chunks = []
        gt_chunks = []

def createGeneratorResizingImages(list_json_files, patch_height, patch_width, batch_size, with_color, considered_classes, reduction_GT):
    
    gr_chunks = []
    gt_chunks = []

    count = 0

    while (True):
        rd.shuffle(list_json_files)

        for json_file_selected in list_json_files:

            src_file_selected = json_file_selected.replace("JSON/", "SRC/").replace(".json", "").replace(".dict", ".png")

            gr, gt = get_image_with_gt(src_file_selected, json_file_selected, with_color, considered_classes, reduction_GT)

            gr = resizeImage(gr, patch_height, patch_width)
            gt = resizeImage(gt, patch_height, patch_width)

            gr_chunks.append(gr)
            gt_chunks.append(gt)

            count +=1
            if count % batch_size == 0:
                gr_chunks_arr = np.array(gr_chunks)
                gt_chunks_arr = np.array(gt_chunks)
                # convert gr_chunks and gt_chunks to the numpy arrays that are yield below

                yield gr_chunks_arr, gt_chunks_arr  # convert into npy before yielding
                gr_chunks = []
                gt_chunks = []
                count = 0


def createGeneratorShuffle(list_json_files, patch_height, patch_width, batch_size, with_color, considered_classes, reduction_GT):
    print("Creating shuffle generator...")
    
    list_shuffle_idx_files = list(range(len(list_json_files)))
        
    while True:
        rd.shuffle(list_shuffle_idx_files)
        for idx_file in list_shuffle_idx_files:
            return createGeneratorResizingImages(list_json_files, patch_height, patch_width, batch_size, with_color, considered_classes, reduction_GT)


def createGeneratorTest(list_json_files, patch_height, patch_width, batch_size, with_color, considered_classes, reduction_GT):
    print("Creating test generator...")
        
    return createGeneratorResizingImagesTest(list_json_files, patch_height, patch_width, batch_size, with_color, considered_classes, reduction_GT)


def appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks):
    gr_sample = gr[
            row : row + patch_height, col : col + patch_width
        ]  # Greyscale image
    gt_sample = gt[
        row : row + patch_height, col : col + patch_width
    ]  # Ground truth
    gr_chunks.append(gr_sample)
    gt_chunks.append(gt_sample)


def getRandomSamplesFromJSONRegions(im_src, symbol_labels, regions_json, n_samples, height_sample, width_sample, coef=1.0):
    X_doc = []
    Y_doc = []

    [ROWS, COLS, _] = get3DShapeImage(im_src)

    if (ROWS < height_sample*coef or COLS < width_sample*coef):
        return [X_doc, Y_doc]

    return regions_json.getRandomSymbolSamples(im_src, symbol_labels, n_samples, height_sample, width_sample)


def getRandomRegionSamplesFromParams(
                    fold,
                    folds,
                    db_names, 
                    n_samples_train, 
                    n_samples_val,
                    n_samples_test,
                    block_size,
                    with_color, 
                    considered_region_class,
                    equalization_mode,
                    with_data_augmentation,
                    reshape):
    assert( type(db_names) is list)

    X_train=[]
    Y_train=[]
    X_val=[]
    Y_val=[]
    X_test=[]
    Y_test=[]

    count = 0

    for db_name_i in db_names:

        [X_train_i, Y_train_i, X_val_i, Y_val_i, X_test_i, Y_test_i] = getRandomRegionSamplesFromParams_1DB(
                    fold=fold, folds=folds, 
                    db_name=db_name_i, 
                    n_samples_train=n_samples_train, n_samples_val=n_samples_val, n_samples_test=n_samples_test,
                    block_size=block_size,
                    with_color=with_color, 
                    considered_class=considered_region_class,
                    equalization_mode=equalization_mode,
                    with_data_augmentation=with_data_augmentation,
                    reshape=reshape)

        if count == 0:
            X_train = X_train_i
            Y_train = Y_train_i
            X_val = X_val_i
            Y_val = Y_val_i
            X_test = X_test_i
            Y_test = Y_test_i
        else:
            X_train = np.concatenate((X_train, X_train_i))
            Y_train = np.concatenate((Y_train, Y_train_i))
            X_val = np.concatenate((X_val, X_val_i))
            Y_val = np.concatenate((Y_val, Y_val_i))
            X_test = np.concatenate((X_test, X_test_i))
            Y_test = np.concatenate((Y_test, Y_test_i))

        count = count + 1

    return [
                X_train, Y_train,
                X_val, Y_val,
                X_test, Y_test
            ]



def get_list_documents_enough_big(list_documents, height_sample, width_sample, coef=1.0):
    list_documents_result = []

    for document in list_documents:
        src_im = FileManager.loadImage(document.src_pathfile, False)
        [ROWS, COLS, _] = get3DShapeImage(src_im)
        src_im = None

        if (ROWS >= height_sample*coef or COLS >= width_sample*coef):
            list_documents_result.append(document)
            
    return list_documents_result
            

def getRandomSamplesFromImage(im_src, im_gt, n_samples, height_sample, width_sample, coef=1.0):

    X_doc = []
    Y_doc = []

    [ROWS, COLS, _] = get3DShapeImage(im_src)

    if (ROWS < height_sample*coef or COLS < width_sample*coef):
        return [X_doc, Y_doc]

    for _ in range (n_samples):
        row = np.random.randint(0, ROWS - height_sample-1)
        col = np.random.randint(0, COLS - width_sample-1)

        sample_src = im_src[row: row + height_sample, col: col + width_sample]
        sample_gt = im_gt[row: row + height_sample, col: col + width_sample]

        X_doc.append(sample_src)
        Y_doc.append(sample_gt)

    return [X_doc, Y_doc]



def lst_pathfiles(path_dir_src, path_dir_gt=None, path_dir_json=None, considered_class=None):
    documents = []

    list_pathfiles_src_all = FileManager.listFilesRecursive(path_dir_src)

    if path_dir_gt is not None:
        list_pathfiles_src = []
        list_pathfiles_gt = []
        list_pathfiles_gt_considered_class = []

        try:
            list_pathfiles_gt_all = FileManager.listFilesRecursive(path_dir_gt)
            if considered_class is not None:
                for pathfile_gt in list_pathfiles_gt_all:
                    if (".JPG_" + considered_class + ".png") in pathfile_gt or (".jpg_" + considered_class + ".png") in pathfile_gt:
                        list_pathfiles_gt_considered_class.append(pathfile_gt)
            assert(len(list_pathfiles_gt_considered_class) <= len(list_pathfiles_src_all))
        except:
            list_pathfiles_gt_all = []
        
        for idx in range (0, len(list_pathfiles_src_all)):
            pathfile_src = list_pathfiles_src_all[idx]

            if list_pathfiles_gt_considered_class is not None:
                for idx_gt in range(0, len(list_pathfiles_gt_considered_class)):

                    pathfile_gt = list_pathfiles_gt_considered_class[idx_gt]

                    if considered_class is not None:
                        aux = pathfile_gt.replace("/GT/", "/SRC/")

                        if (pathfile_src in aux):
                            list_pathfiles_src.append(pathfile_src)
                            list_pathfiles_gt.append(pathfile_gt)
                            break
            else:
                list_pathfiles_src.append(pathfile_src)

        if list_pathfiles_gt_considered_class is not None:
            assert(len(list_pathfiles_gt) == len(list_pathfiles_src))
    else:
        list_pathfiles_src = list_pathfiles_src_all
    
    if path_dir_json is not None:
        list_pathfiles_json = FileManager.listFilesRecursive(path_dir_json)
    
    if path_dir_json is not None:
        assert(len(list_pathfiles_json) == len(list_pathfiles_src))

    for i in range(len(list_pathfiles_src)):
        pathfile_src = list_pathfiles_src[i]
        
        if path_dir_gt is not None:
            pathfile_gt = list_pathfiles_gt[i]
        else:
            pathfile_gt = None
            
        if path_dir_json is not None:
            pathfile_json = list_pathfiles_json[i]
        else:
            pathfile_json = None
        doc = Document(pathfile_src, pathfile_gt, pathfile_json)
        documents.append(doc)

    return documents



def prepareFolds(all_documents, folds, num_test_documents, num_val_documents):

    assert (type(all_documents) is list)
    training_files_folds = []
    test_files_folds = []
    validation_files_folds = []
    
    for idx_fold in range(folds):
        
        idx_test_initial = int(idx_fold * num_test_documents)
        idx_test_end = int(idx_test_initial + num_test_documents)
        
        test_files_folds.append(all_documents[:][idx_test_initial:idx_test_end])
        
        
        idx_val_initial = idx_test_end % len(all_documents)
        idx_val_end = (idx_val_initial + num_val_documents) % len(all_documents)

        if idx_val_end < idx_val_initial:
            idx_val_initial = 0
            idx_val_end = (idx_val_initial + num_val_documents)

        validation_files_folds.append(all_documents[:][idx_val_initial:idx_val_end])
        
        training_aux = []
        for idx_file in range(len(all_documents)):
            
            if idx_file not in range(idx_test_initial, idx_test_end) and idx_file not in range(idx_val_initial, idx_val_end):
                training_aux.append(all_documents[:][idx_file])
                    
        training_files_folds.append(training_aux)

    return [training_files_folds, test_files_folds, validation_files_folds]
    


def lst_pathfiles_folds(path_dir_src, path_dir_gt, path_dir_json, folds, rate_test_documents, rate_val_documents, considered_class=None):

    assert(rate_test_documents>=0 and rate_test_documents < 1)
    assert(rate_val_documents>=0 and rate_val_documents < 1)
    assert(considered_class is None or type(considered_class) is str)
    
    seed = 5

    documents = lst_pathfiles(path_dir_src, path_dir_gt, path_dir_json, considered_class)

    np.random.seed(seed)
    np.random.shuffle(documents)

    num_val_documents = int(rate_val_documents * len(documents))
    num_test_documents = int(rate_test_documents * len(documents))

    [training_docs_folds, test_docs_folds, validation_docs_folds] = prepareFolds(documents, folds, num_test_documents, num_val_documents)

    return [training_docs_folds, test_docs_folds, validation_docs_folds]



def getDocumentsForMURETRegionDetection(
                    fold,
                    folds,
                    db_names, 
                    block_size,
                    with_color,
                    equalization_mode,
                    with_data_augmentation,
                    considered_class):

    docs_train = []
    docs_val = []
    docs_test = []

    count = 0
    for db_name_i in db_names:

        [docs_train_i, docs_val_i, docs_test_i] = getRDocumentsRForMURETRegionDetection_1DB(
                    fold=fold, folds=folds, 
                    db_name=db_name_i, 
                    block_size=block_size,
                    with_color=with_color, 
                    equalization_mode=equalization_mode,
                    with_data_augmentation=with_data_augmentation,
                    considered_class = considered_class)

        if count == 0:
            docs_train = docs_train_i
            docs_val = docs_val_i
            docs_test = docs_test_i
            count = 1
        else:
            docs_train = np.concatenate((docs_train, docs_train_i))
            docs_val = np.concatenate((docs_val, docs_val_i))
            docs_test = np.concatenate((docs_test, docs_test_i))

    return [docs_train, docs_val, docs_test]


def getRandomSampleRegionsDataSetWithJSON(list_documents, num_samples, with_color, equalization_mode, with_data_augmentation, block_size, reshape=None):
    num_documents = len(list_documents)

    num_samples_per_doc = int(num_samples / num_documents)

    X = []
    Y = []

    for document in list_documents:
        
        src_im = FileManager.loadImage(document.src_pathfile, with_color)
        src_im = apply_equalization(src_im, equalization_mode)

        gt_im = FileManager.loadImage(document.gt_pathfile, False)
        gt_im = np.uint8(gt_im > 0)
        
        assert(np.min(gt_im) >= 0 and np.max(gt_im) <=1)
        gt_im = gt_im*255

        if reshape is not None:
            
            if (reshape == 0):
                print ("Rescaling document to the sample size: " + document.src_pathfile)
                src_im = redimImage(src_im, block_size[0], block_size[1])
                gt_im = redimImage(gt_im, block_size[0], block_size[1])
                X_doc =[src_im]
                Y_doc = [gt_im]
            else:
                print ("Rescaling document to selected scale: " + document.src_pathfile)
                src_im = scaleImage(src_im, reshape)
                gt_im = scaleImage(gt_im, reshape)
                [X_doc, Y_doc] = getRandomSamplesFromImage(src_im, gt_im, num_samples_per_doc, block_size[0], block_size[1])
                assert(len(X_doc) <= num_samples_per_doc)
        
        else:
            [X_doc, Y_doc] = getRandomSamplesFromImage(src_im, gt_im, num_samples_per_doc, block_size[0], block_size[1])
            assert(len(X_doc) <= num_samples_per_doc)
            
        assert(len(X_doc) == len(Y_doc))

        for idx in range(len(X_doc)):
            X.append(X_doc[idx])
            Y.append(Y_doc[idx])

    X = np.asarray(X).reshape(len(X), block_size[0], block_size[1], block_size[2])
    Y = np.asarray(Y).reshape(len(Y), block_size[0], block_size[1], 1)

    assert(len(X) > 0)
    if with_data_augmentation:
        [X, Y] = applyDataAugmentation(X=X, Y=Y, return_generator=False)

    return [X, Y]



def getDataSet(list_documents, num_samples, with_color, equalization_mode, with_data_augmentation, block_size):

    coef = 1.0
    list_documents_enough_big = get_list_documents_enough_big(list_documents, block_size[0], block_size[1], coef)
    num_documents = len(list_documents_enough_big)

    num_samples_per_doc = int(num_samples / num_documents)

    X = []
    Y = [] 

    for document in list_documents_enough_big:
        
        src_im = FileManager.loadImage(document.src_pathfile, with_color)
        src_im = apply_equalization(src_im, equalization_mode)

        gt_im = FileManager.loadImage(document.gt_pathfile, False)

        [X_doc, Y_doc] = getRandomSamplesFromImage(src_im, gt_im, num_samples_per_doc, block_size[0], block_size[1], coef)
        assert(len(X_doc) == len(Y_doc))
        assert(len(X_doc) <= num_samples_per_doc)

        for idx in range(len(X_doc)):
            X.append(X_doc[idx])
            Y.append(Y_doc[idx])

    X = np.asarray(X).reshape(len(X), block_size[0], block_size[1], block_size[2])
    Y = np.asarray(Y).reshape(len(Y), block_size[0], block_size[1], 1)

    if with_data_augmentation:
        [X, Y] = applyDataAugmentation(X=X, Y=Y, return_generator=False)

    return [X, Y]


def getListRegionNamesFromJSON(list_documents):
    list_labels = []
    
    for document in list_documents:
        js = CustomJson()
        js.loadJson(document.json_pathfile)

        gt_regions = GTJSONReaderMuret()
        gt_regions.load(js)

        list_labels_doc = gt_regions.getListRegionNames()
        
        list_labels = list(set().union(list_labels, list_labels_doc))


    return list_labels


def getConsideredRegionsFromParamsJSON_ListDB(db_names, fold, folds):
    assert(type(db_names) is list)

    list_labels = []
    
    for db_name_i in db_names:
        list_labels_i = getConsideredRegionsFromParamsJSON(db_name_i, fold, folds)
        list_labels = list(set().union(list_labels, list_labels_i))

    return list_labels


def getConsideredRegionsFromParamsJSON(db_name, fold, folds):

    [pathdir_src_files, pathdir_json_files] = getPathdirDatabaseSRCAndJSON_MURET(db_name)

    [training_docs_folds, _, _] = lst_pathfiles_folds(
                    path_dir_src=pathdir_src_files, 
                    path_dir_gt=None, 
                    path_dir_json=pathdir_json_files, 
                    folds=folds, 
                    rate_test_documents=0.2, 
                    rate_val_documents=0.2)

    return getListRegionNamesFromJSON(training_docs_folds[fold])

        
def getRDocumentsRForMURETRegionDetection_1DB(
                    fold, folds, 
                    db_name, 
                    block_size,
                    with_color, 
                    equalization_mode,
                    with_data_augmentation,
                    considered_class):
    [pathdir_src_files, pathdir_gt_files, pathdir_json_files] = getPathdirDatabaseSRCAndJSON_MURET(db_name=db_name, with_gt=True)

    if FileManager.existsFile(pathdir_gt_files) == False:
        pathdir_gt_files_aux = None
    else:
        pathdir_gt_files_aux = pathdir_gt_files

    [training_docs_folds, test_docs_folds, validation_docs_folds] = lst_pathfiles_folds(
                    path_dir_src=pathdir_src_files, 
                    path_dir_gt=pathdir_gt_files_aux, 
                    path_dir_json=pathdir_json_files, 
                    folds=folds, 
                    rate_test_documents=0.2, 
                    rate_val_documents=0.2,
                    considered_class=considered_class)

    return [training_docs_folds[fold], test_docs_folds[fold], validation_docs_folds[fold]]

def getRandomRegionSamplesFromParams_1DB(
                    fold, folds, 
                    db_name, 
                    n_samples_train, n_samples_val, n_samples_test,
                    block_size,
                    with_color, 
                    considered_class,
                    equalization_mode,
                    with_data_augmentation,
                    reshape):
    assert(type(db_name) is str)
    assert(type(considered_class) is str)

    [pathdir_src_files, pathdir_gt_files, pathdir_json_files] = getPathdirDatabaseSRCAndJSON_MURET(db_name=db_name, with_gt=True)

    [training_docs_folds, test_docs_folds, validation_docs_folds] = lst_pathfiles_folds(
                    path_dir_src=pathdir_src_files, 
                    path_dir_gt=pathdir_gt_files, 
                    path_dir_json=None, 
                    folds=folds, 
                    rate_test_documents=0.2, 
                    rate_val_documents=0.2,
                    considered_class=considered_class)

    [X_train, Y_train] = getRandomSampleRegionsDataSetWithJSON(
                    list_documents=training_docs_folds[fold],
                    num_samples=n_samples_train, 
                    with_color=with_color, 
                    equalization_mode=equalization_mode, 
                    with_data_augmentation=with_data_augmentation, 
                    block_size=block_size,
                    reshape=reshape)

    [X_val, Y_val] = getRandomSampleRegionsDataSetWithJSON(
                    list_documents=validation_docs_folds[fold], 
                    num_samples=n_samples_val, 
                    with_color=with_color, 
                    equalization_mode=equalization_mode, 
                    with_data_augmentation=with_data_augmentation, 
                    block_size=block_size,
                    reshape=reshape)

    [X_test, Y_test] = getRandomSampleRegionsDataSetWithJSON(
                    list_documents=test_docs_folds[fold], 
                    num_samples=n_samples_test, 
                    with_color=with_color, 
                    equalization_mode=equalization_mode, 
                    with_data_augmentation=with_data_augmentation, 
                    block_size=block_size,
                    reshape=reshape)

    return [
                X_train, Y_train,
                X_val, Y_val,
                X_test, Y_test
            ]

def applyDataAugmentation(
                    X, Y, 
                    rotation_range=8, 
                    shear_range=0, 
                    vertical_flip=False, 
                    horizontal_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    fill_mode='nearest',
                    cval=0,
                    batch_size=32,
                    save_images=False, pathfile_saved_images=None,
                    return_generator=False):
    data_gen_args = dict(
                    rotation_range=rotation_range,              #3   10,           # Int. Degree range for random rotations.
                    shear_range=shear_range,                    # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
                    vertical_flip=vertical_flip,                # Boolean. Randomly flip inputs vertically.
                    horizontal_flip=horizontal_flip,
                    width_shift_range=width_shift_range,        #0.08, 0.09
                    height_shift_range=height_shift_range,      #0.08,  0.09
                    zoom_range=zoom_range,                      #0.08, 0.09                # Float or [lower, upper]. Range for random zoom. If a float,
                    fill_mode=fill_mode,
                    cval=cval
                    #brightness_range=[0.2, 1.4]  # Tuple or list of two floats. Range for picking a brightness shift value from.
                    )

    data_gen_seg_args = dict(
                    rotation_range=rotation_range,              #3   10,           # Int. Degree range for random rotations.
                    shear_range=shear_range,                    # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
                    vertical_flip=vertical_flip,                # Boolean. Randomly flip inputs vertically.
                    horizontal_flip=horizontal_flip,
                    width_shift_range=width_shift_range,        #0.08, 0.09
                    height_shift_range=height_shift_range,      #0.08,  0.09
                    zoom_range=zoom_range,                      #0.08, 0.09                # Float or [lower, upper]. Range for random zoom. If a float,
                    fill_mode='constant',
                    cval=255
                    #brightness_range=[0.2, 1.4]  # Tuple or list of two floats. Range for picking a brightness shift value from.
                    )

    num_samples = len(X)
    image_datagen = ImageDataGenerator(**data_gen_args)
    seg_datagen = ImageDataGenerator(**data_gen_seg_args)

    save_params1 = dict()
    save_params2 = dict()
    if save_images:
        assert(pathfile_saved_images is not None)
        out_x = pathfile_saved_images + "_src"
        out_s = pathfile_saved_images + "_gt"
        save_params1 = dict(save_to_dir=out_x, save_prefix='IM_plant', save_format='png')
        save_params2 = dict(save_to_dir=out_s, save_prefix='seg', save_format='png')

    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=1, **save_params1)
    seg_generator = seg_datagen.flow(Y, batch_size=batch_size, seed=1, **save_params2)

    def combine_generator(gen1, gen2):
        while True:
            yield(gen1.next(), gen2.next())

    if return_generator:
        return combine_generator(image_generator, seg_generator)
    else:
        X = []
        Y = []
        count = 0
        for X_new, Y_new in combine_generator(image_generator, seg_generator):

            X_new = X_new.astype(np.uint8)
            Y_new = Y_new.astype(np.uint8)
            if (count == 0):
                X = X_new
                Y = Y_new
            else:
                X = np.concatenate((X, X_new))
                Y = np.concatenate((Y, Y_new))
            
            count = count + len(Y_new) 
            if (count > num_samples):
                break

        return [X, Y]




