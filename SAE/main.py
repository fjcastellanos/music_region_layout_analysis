
import argparse
import os, sys
import time
from utils.ImageProcessing import EQUALIZATION_TYPES
from utils.ModelConfig import NORMALIZATION_LIST, ModelConfig, TYPE_NORMALIZATION_STANDARD, TYPE_NORMALIZATION_MEAN
from utils.Databases import MURET_DATABASE_LIST, getPathdirDatabase, getFullPathParentDatabasesContainer
from utils.DataLoader import *
from utils.GTJSONReaderMuret import *
from utils.CustomJson import CustomJson
from utils.file_manager import FileManager
from utils.RegionExtraction import *
import json

opt_list = ['sgd', 'adam', 'adadelta']

KEY_MODE_TRAIN = 'train'
KEY_MODE_TEST = 'test'
KEY_MODE_TEST_FULL_PAGES = 'test-pg'
exec_mode_list = [KEY_MODE_TRAIN, KEY_MODE_TEST, KEY_MODE_TEST_FULL_PAGES]

KEY_MODEL = 'mdl'
KEY_MODE = 'mde'
KEY_GPU = 'gpu'
KEY_COLOR_MODE = 'cm'
KEY_SAMPLE_WIDTH = 'sw'
KEY_SAMPLE_HEIGHT = 'sh'
KEY_KERNEL_WIDTH = 'kw'
KEY_KERNEL_HEIGHT = 'kh'
KEY_NUMBER_FILTERS = 'nfl'
KEY_BATCH_SIZE = 'b'
KEY_NORMALIZATION = 'n'
KEY_EPOCHS = 'e'
KEY_SAVE_RESULT_IMAGES = 'i'
KEY_SAVE_MODEL_GRAPH = 'gp'
KEY_LABELS = 'lb'


KEY_SAE = 'sae'
KEY_DATABASE_TRAIN = 'db_tr'
KEY_DATABASE_VAL = 'db_v'
KEY_DATABASE_TEST = 'db_ts'

KEY_NUM_BLOCKS = "nb"
KEY_WITH_POSTPROCESSING = 'p'

KEY_GENERATE_GT = 'gt'
KEY_FSCORE_THRESHOLD = 'th'
KEY_GT_HEIGHT_REDUCTION = "red"
KEY_DATABASE_GENERATING_GT= "db-gt"
KEY_NUMBER_IMAGES_CONSIDERED = "nimgs"
KEY_OUTPUT_RESULTS_TEST = "results_out"

kCONSIDERED_CLASSES = {
            "staff": ["staff", "empty_staff"],
            "lyrics": ["lyrics"], 
            "multiple_lyrics": ["multiple_lyrics"], 
            "text":["text"],
            "title":["title"],
            "author":["author"],
            "undefined":["undefined"]
            #"pages":["pages"]
    }

def addGenericArguments(parser):
    parser.add_argument('--mode',       dest=KEY_MODE,                      help='Execution mode', choices=exec_mode_list, required=True)
    parser.add_argument('--gpu',        dest=KEY_GPU,                       help='Identifier of GPU', type=int, required=True)
    parser.add_argument('--cmode',      dest=KEY_COLOR_MODE,                help='Color mode', type=int, required=True)
    parser.add_argument('--s-width',    dest=KEY_SAMPLE_WIDTH,              help='Sample width', type=int, required=True)
    parser.add_argument('--s-height',   dest=KEY_SAMPLE_HEIGHT,             help='Sample height', type=int, required=True)
    parser.add_argument('--k-height',   dest=KEY_KERNEL_HEIGHT,             help='Kernel height', type=int, required=False, default=3)
    parser.add_argument('--k-width',    dest=KEY_KERNEL_WIDTH,              help='Kernel width', type=int, required=False, default=3)
    parser.add_argument('--nfilt',      dest=KEY_NUMBER_FILTERS,            help='Number of filters', type=int, required=False, default=128)
    parser.add_argument('--batch',      dest=KEY_BATCH_SIZE,                help='Batch size', type=int, required=True)
    parser.add_argument('--norm',       dest=KEY_NORMALIZATION,             help='Type of normalization', choices=NORMALIZATION_LIST, required=True)
    parser.add_argument('--epochs',     dest=KEY_EPOCHS,                    help='Maximum number of epochs', type=int, required=True)
    parser.add_argument('--nbl',        dest=KEY_NUM_BLOCKS,                help='Number of blocks in the encoder and decoder of the SAE model', type=int, required=False, default=3)
    parser.add_argument('--img',        dest=KEY_SAVE_RESULT_IMAGES,        help='Save images', action='store_true', required=False)
    parser.add_argument('--graph',      dest=KEY_SAVE_MODEL_GRAPH,          help='Save model graph', action='store_true', required=False)
    parser.add_argument('--post',       dest=KEY_WITH_POSTPROCESSING,       help='With post-processing', action='store_true', required=False)
    parser.add_argument('--th',         dest=KEY_FSCORE_THRESHOLD,          help='Threshold for region fscore IoU', type=float, required=False)
    parser.add_argument('--nimgs',      dest=KEY_NUMBER_IMAGES_CONSIDERED,  help='Number of images considered from the training set', type=int, required=True)
    parser.add_argument('--out',        dest=KEY_OUTPUT_RESULTS_TEST,       help='Path folder for the results', default="Output")
    
    
    

def addArgumentSAE(parsers):
    parser = parsers.add_parser(KEY_SAE)

    parser.add_argument('--db-train',   dest=KEY_DATABASE_TRAIN,       help='Database for training', required=True)
    parser.add_argument('--db-val',     dest=KEY_DATABASE_VAL,         help='Database for validation', required=True)
    parser.add_argument('--db-test',    dest=KEY_DATABASE_TEST,        help='Database for validation', required=False)
    parser.add_argument('--labels',     dest=KEY_LABELS,         action="append",      help='Considered labels',        required=False)
    parser.add_argument('--red',        dest=KEY_GT_HEIGHT_REDUCTION,          help='Reduction factor for ground truth data', type=float, required=False, default=0.)


    addGenericArguments(parser)


def addGenerateGTfromJSON(parsers):
    parser = parsers.add_parser(KEY_GENERATE_GT)
    parser.add_argument('--red',         dest=KEY_GT_HEIGHT_REDUCTION,          help='Reduction factor for ground truth data', type=float, required=False)
    parser.add_argument('--dataset',     dest=KEY_DATABASE_GENERATING_GT,       help='Database for generating GT', required=False)



def saveInputConfig(parsed_args, pathfile):
    FileManager.saveDictionary(parsed_args, pathfile)


def getFileName_params(label, parsed_args):

    parsed_args_without_extra_data = parsed_args.copy()
    parsed_args_without_extra_data.pop(KEY_DATABASE_TEST, None) 
    parsed_args_without_extra_data.pop(KEY_WITH_POSTPROCESSING, None)
    parsed_args_without_extra_data.pop(KEY_GPU, None) 
    parsed_args_without_extra_data.pop(KEY_SAVE_RESULT_IMAGES, None) 
    parsed_args_without_extra_data.pop(KEY_SAVE_MODEL_GRAPH, None) 
    parsed_args_without_extra_data.pop(KEY_MODE, None)
    parsed_args_without_extra_data.pop(KEY_FSCORE_THRESHOLD, None)
    parsed_args_without_extra_data.pop(KEY_LABELS, None)
    parsed_args_without_extra_data.pop(KEY_OUTPUT_RESULTS_TEST, None)
    
    pathfile = str(sorted(parsed_args_without_extra_data.items()))
    pathfile = pathfile.replace("[(", "").replace(")]", "").replace("OrderedDict", "").replace(", ", "-").replace("(", "_").replace(")", "").replace("'", "").replace("-_", "_")
    pathfile = pathfile.replace("{", "").replace("}", "").replace("': ", "").replace("'", "").replace(", ", "_")
    pathfile = pathfile + "_" + label
    pathfile = pathfile.replace(".", "_")

    return pathfile




def get_pathfile_saved_model_params(label, parsed_args):

    parsed_args_without_test_data = parsed_args.copy()
    del parsed_args_without_test_data[KEY_MODE]
    del parsed_args_without_test_data[KEY_GPU]
    parsed_args_without_test_data.pop(KEY_DATABASE_TEST, None) 
    parsed_args_without_test_data.pop(KEY_WITH_POSTPROCESSING, None) 
    parsed_args_without_test_data.pop(KEY_SAVE_RESULT_IMAGES, None) 
    parsed_args_without_test_data.pop(KEY_SAVE_MODEL_GRAPH, None) 
    parsed_args_without_test_data.pop(KEY_FSCORE_THRESHOLD, None)
    parsed_args_without_test_data.pop(KEY_LABELS, None)
    parsed_args_without_test_data.pop(KEY_OUTPUT_RESULTS_TEST, None)

    if KEY_GT_HEIGHT_REDUCTION in parsed_args_without_test_data and parsed_args_without_test_data[KEY_GT_HEIGHT_REDUCTION] == 0:
        parsed_args_without_test_data.pop(KEY_GT_HEIGHT_REDUCTION, None)

    fileName = getFileName_params(label, parsed_args_without_test_data)
    pathfile = "saved_models/%s/%s.h5" %\
                        (
                        parsed_args_without_test_data[KEY_MODEL],
                        fileName)

    createDirsIfNeed(pathfile=pathfile, parsed_args=parsed_args_without_test_data, save_config=False)
    return pathfile


def createDirsIfNeed(pathfile, parsed_args, save_config = True):
    pathdir = FileManager.nameOfDirFromPath(pathfile)
    FileManager.makeDirsIfNeeded(pathdir)

    if save_config:
        FileManager.saveDictionary(parsed_args, pathdir+"/config.json")

def config_get_kernel_shape(parsed_args):
    return (parsed_args[KEY_KERNEL_HEIGHT], parsed_args[KEY_KERNEL_WIDTH])

def config_get_number_images_considered(parsed_args):
    return parsed_args[KEY_NUMBER_IMAGES_CONSIDERED]


def config_gt_reduction(parsed_args):
    return parsed_args[KEY_GT_HEIGHT_REDUCTION]

def config_get_threshold_for_fscore(parsed_args):
    if KEY_FSCORE_THRESHOLD in parsed_args and parsed_args[KEY_FSCORE_THRESHOLD] is not None:
        return parsed_args[KEY_FSCORE_THRESHOLD]
    else:
        return 0.55

def config_get_number_filters(parsed_args):
    return parsed_args[KEY_NUMBER_FILTERS]


def config_get_num_blocks(parsed_args):
    return parsed_args[KEY_NUM_BLOCKS]

def config_with_color(parsed_args):
    return bool(parsed_args[KEY_COLOR_MODE])


def config_get_batch_size(parsed_args):
    return parsed_args[KEY_BATCH_SIZE]

def config_with_saving_images(parsed_args):
    return parsed_args[KEY_SAVE_RESULT_IMAGES]
        
def config_with_saving_model_graph(parsed_args):
    return parsed_args[KEY_SAVE_MODEL_GRAPH]


def get_model_config(parsed_args):
    return ModelConfig(
                    type_normalization=parsed_args[KEY_NORMALIZATION], 
                    epochs=parsed_args[KEY_EPOCHS],
                    batch_size=parsed_args[KEY_BATCH_SIZE])


def getConsisderedClassesAllDatabases(db_names=MURET_DATABASE_LIST):
    
    considered_classes = kCONSIDERED_CLASSES
    print(considered_classes)

    return considered_classes


def testSAE(parsed_args, bn_axis):
    block_size = getBlockSize(parsed_args)
    with_color = config_with_color(parsed_args)
    num_blocks = config_get_num_blocks(parsed_args)
    with_saving_images = config_with_saving_images(parsed_args)
    kernel_shape = config_get_kernel_shape(parsed_args)
    n_filters = config_get_number_filters(parsed_args)
    batch_size = config_get_batch_size(parsed_args)
    number_images_considered = config_get_number_images_considered(parsed_args)
    reduction_GT = config_gt_reduction(parsed_args)

    considered_classes = getConsideredClasses(parsed_args)

    for considered_class in considered_classes:

        real_classes = getConsisderedClassesAllDatabases(db_names = parsed_args[KEY_DATABASE_TRAIN])
        assert(considered_class in real_classes)

        pathfile_saved_model=get_pathfile_saved_model_params(label=considered_class, parsed_args=parsed_args)


        list_train_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_TRAIN])
        list_train_files = sorted(list_train_files)
        list_train_files = list_train_files[0:number_images_considered]

        list_val_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_VAL])
        list_test_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_TEST])
        
        sae = SAE(
                input_shape=block_size,
                kernel_shape=kernel_shape,
                n_filters=n_filters,
                pool=2,
                bn_axis=bn_axis,
                activation_function='relu',
                with_batch_normalization=False,
                num_blocks=num_blocks,
                with_saving_images = with_saving_images,
                pathfile_saved_model=pathfile_saved_model)


        sae.load_model()

        generator_test = createGeneratorTest(
                list_json_files = list_test_files, 
                patch_height = block_size[0], 
                patch_width = block_size[1],
                batch_size = 1,
                with_color=with_color,
                considered_classes = considered_classes[considered_class],
                reduction_GT=reduction_GT
                )

        predictions = sae.model.predict(generator_test)

        idx_file = 0
        for prediction in predictions:
            pred_2D = np.argmax(prediction, axis=2)

            json_test_file = list_test_files[idx_file]
            src_test_file = json_test_file.replace("JSON/", "SRC/").replace(".json", "")
            src_img = FileManager.loadImage(src_test_file, False)

            pred_2D_resized = resizeImage(pred_2D*255., src_img.shape[0], src_img.shape[1])

            filename = os.path.basename(src_test_file)
            parent_path_folder = parsed_args[KEY_OUTPUT_RESULTS_TEST]

            pathfile_out = os.path.join(parent_path_folder, filename)
            pathfile_out = os.path.join(parent_path_folder, "IMAGES", filename)

            pathfile_out_json = os.path.join(parent_path_folder, filename)
            pathfile_out_json = os.path.join(parent_path_folder, "JSON", filename)
            pathfile_out_json = pathfile_out_json + ".json"

            bboxes = getBoundingBoxes(pred_2D_resized, val=100)

            FileManager.saveImageFullPath((pred_2D)*255, pathfile_out)
            
            content_string_json = json.dumps(bboxes, indent=2)
            FileManager.saveString(content_string_json, pathfile_out_json, True)

            idx_file += 1



        print("")

def trainSAE(parsed_args, bn_axis):
    block_size = getBlockSize(parsed_args)
    with_color = config_with_color(parsed_args)
    num_blocks = config_get_num_blocks(parsed_args)
    with_saving_images = config_with_saving_images(parsed_args)
    kernel_shape = config_get_kernel_shape(parsed_args)
    n_filters = config_get_number_filters(parsed_args)
    batch_size = config_get_batch_size(parsed_args)
    number_images_considered = config_get_number_images_considered(parsed_args)
    reduction_GT = config_gt_reduction(parsed_args)

    considered_classes = getConsideredClasses(parsed_args)

    for considered_class in considered_classes:

        real_classes = getConsisderedClassesAllDatabases(db_names = parsed_args[KEY_DATABASE_TRAIN])
        assert(considered_class in real_classes)

        pathfile_saved_model=get_pathfile_saved_model_params(label=considered_class, parsed_args=parsed_args)

        if ".txt" in parsed_args[KEY_DATABASE_TRAIN]:
            list_train_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_TRAIN])
        else:
            list_train_files = FileManager.listFilesRecursive(parsed_args[KEY_DATABASE_TRAIN])
            
        print(number_images_considered)
        list_train_files = sorted(list_train_files)

        list_train_files = list_train_files[0:number_images_considered]

        list_val_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_VAL])

        print("TRAIN:")
        print(list_train_files)

        print("VAL:")
        print(list_val_files)

        sae = SAE(
                input_shape=block_size,
                kernel_shape=kernel_shape,
                n_filters=n_filters,
                pool=2,
                bn_axis=bn_axis,
                activation_function='relu',
                with_batch_normalization=False,
                num_blocks=num_blocks,
                with_saving_images = with_saving_images,
                pathfile_saved_model=pathfile_saved_model)

        generator_train = createGeneratorShuffle(
                list_json_files = list_train_files, 
                patch_height = block_size[0], 
                patch_width = block_size[1],
                batch_size = batch_size,
                with_color=with_color,
                considered_classes = considered_classes[considered_class],
                reduction_GT=reduction_GT
                )

        generator_val = createGeneratorShuffle(
                list_json_files = list_val_files, 
                patch_height = block_size[0], 
                patch_width = block_size[1],
                batch_size = batch_size,
                with_color=with_color,
                considered_classes = considered_classes[considered_class],
                reduction_GT=reduction_GT
                )
    
        model_config = get_model_config(parsed_args)

        start_time = time.time()
        sae.train(
                list_json_files_train = list_train_files,
                list_json_files_val = list_val_files,
                generator_train = generator_train,
                generator_val = generator_val,
                model_config=model_config)
        end_time = time.time()

        total_time = end_time - start_time
        print("%.2f seconds" % total_time)



def getConsideredClasses(parsed_args):

    all_considered_classes = getConsisderedClassesAllDatabases(db_names=parsed_args[KEY_DATABASE_TRAIN])

    if KEY_LABELS in parsed_args and parsed_args[KEY_LABELS] is not None and len(parsed_args[KEY_LABELS]) > 0:
        considered_classes = {}
        for label in parsed_args[KEY_LABELS]:
            assert(label in all_considered_classes)
            considered_classes[label] = all_considered_classes[label]
    else:
        considered_classes = all_considered_classes

    assert(len(considered_classes) > 0)

    return considered_classes




def config_with_post_processing(parsed_args):
    if KEY_WITH_POSTPROCESSING in parsed_args and parsed_args[KEY_WITH_POSTPROCESSING] is not None:
        return parsed_args[KEY_WITH_POSTPROCESSING]
    else:
        return False

def getBlockSize(parsed_args):
    with_color = config_with_color(parsed_args)
    channels = 3 if with_color else 1
    return (parsed_args[KEY_SAMPLE_WIDTH], parsed_args[KEY_SAMPLE_HEIGHT], channels)




def generateGTfromJSON(parsed_args):

    gt_dataset_dir = "databases/MURET/GT"
    json_dataset_dir = "databases/MURET/JSON"

    if KEY_DATABASE_GENERATING_GT in parsed_args:
        gt_dataset_dir = gt_dataset_dir + "/" + parsed_args[KEY_DATABASE_GENERATING_GT]
        json_dataset_dir = json_dataset_dir + "/" + parsed_args[KEY_DATABASE_GENERATING_GT]

    FileManager.deleteFolder(gt_dataset_dir)
    json_pathfiles = FileManager.listFilesRecursive(json_dataset_dir)


    import progressbar
        
    considered_classes = kCONSIDERED_CLASSES


    progress_bar = progressbar.ProgressBar(maxval=len(json_pathfiles), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    print ("Generating GT files")
    progress_bar.start()
    
    idx_file = 0
    for json_pathfile in json_pathfiles:
        js = CustomJson()
        js.loadJson(json_pathfile)

        gt_regions = GTJSONReaderMuret()
        gt_regions.load(js)

        for considered_class in considered_classes:
            gt_pathfile = json_pathfile.replace("/JSON/", "/GT/").replace(".json", "_" + str(considered_class) + ".dat")
            src_pathfile = json_pathfile.replace("/JSON/", "/SRC/").replace(".json", "")

            src_im = FileManager.loadImage(src_pathfile, True)

            gt_shape = (src_im.shape[0], src_im.shape[1])
            gt_im = gt_regions.generateGT(considered_classes[considered_class], gt_shape, parsed_args[KEY_GT_HEIGHT_REDUCTION])

            FileManager.saveImageFullPath(gt_im * 255, gt_pathfile)
            #FileManager.saveGTImage2(gt_im, gt_pathfile)

        idx_file = idx_file + 1
        progress_bar.update(idx_file)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=KEY_MODEL, help='Model')

    addGenerateGTfromJSON(subparsers)
    addArgumentSAE(subparsers)

    args = parser.parse_args()
    parsed_args = vars(args)
    print(parsed_args)

    if ".txt" in parsed_args[KEY_DATABASE_TRAIN]:
        list_train_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_TRAIN])
    else:
        list_train_files = FileManager.listFilesRecursive(parsed_args[KEY_DATABASE_TRAIN])
    
    list_val_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_VAL])
    list_test_files = FileManager.readListItemsFromFile(parsed_args[KEY_DATABASE_TEST])


    if parsed_args[KEY_MODEL] == KEY_SAE:

        import os
        if (parsed_args[KEY_GPU] is not None):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(parsed_args[KEY_GPU])
            print ("GPU is set in " + str(parsed_args[KEY_GPU]))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"
            print ("GPU is set in 0")

        from keras import backend as K
        if K.backend() == 'tensorflow':
            import tensorflow as tf    # Memory control with Tensorflow
            try:
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth=True
                session_tf = tf.compat.v1.Session(config=config)
            except:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                session_tf = tf.Session(config=config)
            #K.set_session(session_tf)
        
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        from utils.DataLoader import *
        from utils.SAE import SAE

        if parsed_args[KEY_MODE] == KEY_MODE_TRAIN:
            trainSAE(parsed_args, bn_axis)
        elif parsed_args[KEY_MODE] == KEY_MODE_TEST:
            testSAE(parsed_args, bn_axis)
        else:
            assert(False)

    elif parsed_args[KEY_MODEL] == KEY_GENERATE_GT:
        generateGTfromJSON(parsed_args)
    else:
        assert(False)

    

