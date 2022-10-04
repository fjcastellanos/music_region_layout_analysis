import numpy as np
import os
import cv2 as cv
import torch
import constants as CONSTANTS
import utils
import fiftyone as fo
import evaluation

from models import MODELS, MODELS_FUNC, MODELS_IMAGE_SIZE, EarlyStopping, ModelCheckpoint
from args import args_parser_training
from dataset import MusicDataset, get_transform
from engine import train_one_epoch, evaluate



def get_training_folders(args):
   """Calculates and returns filenames for images and its labels (bounding boxes, classes, etc)

   Args:
      args: dict
         args of the program
      folder: np.array
         contains the folder for each division [training, validation, test]

   Returns:
      training_folders: list
         list with the paths for all json files that contains labels + information
      training_folders_images: list
         list with the paths for all json files that contains images
   """
   training_folder, training_folder_images = [], [None, None, None]
   if args.n_pages is not None and args.n_images is not None:
      print(f'DATA GENERATION TRAINING')
      rotation = CONSTANTS.PREFIX_RND_AUTO if args.unif_rotation == 0 else CONSTANTS.PREFIX_RND_AUTO_UNIF
      training_folder = CONSTANTS.DATA_ROOT_FOLDER + CONSTANTS.FOLDER_DAUG + rotation
      training_folder += str(args.n_pages) + "_pages/" + CONSTANTS.BASE_FOLDER + "fold" + str(args.fold) + "/JSON/" + args.dataset + "/"
      base_path = training_folder
      print(f'Base path for training folder (JSON images) {base_path} ')
      training_folder = [base_path+file for file in sorted(os.listdir(training_folder))]
      # Selecting only the quantity we want (n_images)
      training_folder = training_folder[:args.n_images]
      
      training_folder_images = CONSTANTS.DATA_ROOT_FOLDER + CONSTANTS.FOLDER_DAUG + rotation
      training_folder_images += str(args.n_pages) + "_pages/" + CONSTANTS.BASE_FOLDER + "fold" + str(args.fold) + "/SRC/" + args.dataset + "/"
      base_path = training_folder_images
      print(f'Base path for training folder images {base_path} ')
      training_folder_images = [base_path+file for file in sorted(os.listdir(training_folder_images))]

      # Selecting only the quantity we want (n_images)
      training_folder_images = training_folder_images[:args.n_images]
      training_folder = training_folder[:args.n_images]
      print(f'Training with {len(training_folder_images)} generated images')

   else:
      print(f'NORMAL TRAINING')
      filepath_base = CONSTANTS.DATA_ROOT_FOLDER + 'Folds/' + args.dataset + '/fold'
      training_folder = filepath_base + str(args.fold) + '/train.txt'

      print(f'Path for training {training_folder}')
      with open(training_folder, 'r') as file:
         lines = file.readlines()
         lines[-1] = lines[-1] + '\n'
         training_folder = [lines.replace('\n', '') for lines in lines]
         lines_images = [line.replace('JSON/','').replace('datasets', 'data/SRC')[:-6] for line in lines]
         training_folder_images = lines_images

         # Selecting only the quantity we want (n_images)
         print(f'SELECTING {args.n_images} from training folder')
         training_folder_images = training_folder_images[:args.n_images]
         training_folder = training_folder[:args.n_images]

         # Quit last '.'
         if training_folder_images[-1][-1] == '.':
            training_folder_images[-1] = training_folder_images[-1][:-1]
            
   return training_folder, training_folder_images


def get_eval_folders(args):
   """Calculates and returns filenames for images and its labels (bounding boxes, classes, etc)
   Args:
      args: dict
         args of the program
      folder: np.array
         contains the folder for each division [training, validation, test]

   Returns:
      eval_folders: list
         list with the paths for all json files that contains labels + information
      eval_folders_images: list
         list with the paths for all json files that contains images
   """

   eval_folder, eval_folder_images = [], [None, None, None]
   filepath_base = CONSTANTS.DATA_ROOT_FOLDER + 'Folds/' + args.dataset + '/fold'
   eval_folder = filepath_base + str(args.fold) + '/val.txt'

   print(f'Path for evaluation {eval_folder}')
   with open(eval_folder, 'r') as file:
      lines = file.readlines()
      lines[-1] = lines[-1] + '\n'
      eval_folder = [lines.replace('\n', '') for lines in lines]
      lines_images = [line.replace('JSON/','').replace('datasets', 'data/SRC')[:-6] for line in lines]
      eval_folder_images = lines_images

   return eval_folder, eval_folder_images


def get_model_filepath(args):
   """
      Obtains the filepath name for the current model
      Example1: checkpoint/faster_cnn/no-aug/b-59-850/foldX/model-Nimages.pt 
      Example2: checkpoint/faster_cnn/random-auto/b-59-850/foldX/model-Ypages-Nimages.pt
      Args:
         args: dict
            contains arguments passed to the program

      Returns:
         path_checkpoint: string
            name of the model checkpoint with .pt extensios (PyTorch ext)
   """
   # args [model, unif_rot, n_images, n_pages, dataset, fold] CONSTANTS
   path_checkpoint = CONSTANTS.CHECKPOINT_ROOT + args.model.lower() + "/" 

   if args.unif_rotation is not None:
      aug_folder = CONSTANTS.PREFIX_RND_AUTO if args.unif_rotation == 0 else CONSTANTS.PREFIX_RND_AUTO_UNIF
      pages_images = "-" + str(args.n_pages) + 'pages-' + str(args.n_images) + 'images'
   else:
      print(f'NO AUG MODEL FILEPATH')
      aug_folder = 'no-aug/' 
      pages_images = "-" + str(args.n_images) + 'images'

   path_checkpoint += aug_folder + args.dataset + "/fold" + str(args.fold) + "/model" + pages_images + CONSTANTS.CHECKPOINT_EXTENSION

   return path_checkpoint



if __name__ == "__main__":
   args = args_parser_training.parse_args()
   resized_shape = MODELS_IMAGE_SIZE[args.model]
   print(f'FOLD: {args.fold}')
   print(f'CURRENT PATH: {os.getcwd()}')
   print(f'ARGS: {args}')
   print(f'MODEL SELECTED: {args.model}')
   print(f'SIZE OF IMAGES: {resized_shape}')
   # Gets data for training and evaluation (regardless is generated or not)
   training_folders, training_folders_images = get_training_folders(args)
   eval_folders, eval_folders_images = get_eval_folders(args)

   # Gets data for training and evaluation (regardless is generated or not)
   num_epochs = args.epochs
   
   print(f'Data folders for training: ')
   print(f'Len of training folders: {len(training_folders)}')
   print(f'Len of training folders images: {len(training_folders_images)}')
   print(training_folders, "\n\n", training_folders_images)
   print()

   print(f'Data folders for eval: ')
   print(f'Len of eval folders: {len(eval_folders)}')
   print(f'Len of eval folders images: {len(eval_folders_images)}')
   print(eval_folders, "\n\n", eval_folders_images)

   print(f'Loading torch {args.model}')

   # Load the model
   # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   # num_classes = CONSTANTS.NUM_CLASSES + 1 # Staff, Lyrics + Background
   # in_features = model.roi_heads.box_predictor.cls_score.in_features
   # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

   # model = faster_rcnn()
   # model = get_model(args)
   # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, num_classes=CONSTANTS.NUM_CLASSES+1)

   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   model = MODELS_FUNC[args.model]()
   model.to(device)
   print(f'Loaded pretrained model')
   
   # Load the datasets
   # Training dataset
   batch_size = args.batch_size
   print(f'BATCH SIZE FOR TRAINING: {batch_size}')
   dataset_train = MusicDataset(args.dataset, training_folders, training_folders_images, resized_shape, get_transform())
   data_loader_train = torch.utils.data.DataLoader(
      dataset_train, batch_size=batch_size, shuffle=True, num_workers=1,
      collate_fn=utils.collate_fn
   )

   # Validation  dataset (no redimension in order to calculate metrics)
   dataset_eval = MusicDataset(args.dataset, eval_folders, eval_folders_images, None, get_transform())
   data_loader_eval = torch.utils.data.DataLoader(
      dataset_eval, batch_size=1, shuffle=False, num_workers=2,
      collate_fn=utils.collate_fn
   )

   # Resized for coco (to check if metrics change)
   dataset_eval_coco = MusicDataset(args.dataset, eval_folders, eval_folders_images, resized_shape, get_transform())
   data_loader_eval_coco = torch.utils.data.DataLoader(
      dataset_eval_coco, batch_size=1, shuffle=False, num_workers=2,
      collate_fn=utils.collate_fn
   )

   # Params and optimizer
   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   params = [p for p in model.parameters() if p.requires_grad]
   optimizer = torch.optim.SGD(params, lr=0.001,
                                 momentum=0.9, weight_decay=0.0005)
   lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
   model.train()
   model.to(device)

   path_checkpoint = get_model_filepath(args)
   print(f'Path for model checkpoint: {path_checkpoint}')


   # Callbacks
   modelCheckpoint = ModelCheckpoint(path_checkpoint, model)
   earlyStopping = EarlyStopping(patience=args.patience, epochs=num_epochs, checkpoint=modelCheckpoint, epsilon=0.01)
   
   F1_strings = ['Average F1  @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =',]

   print(f'STARTING TRAINING')
   dataset_eval_fo = fo.Dataset(name="dataset_eval")
   list_images = dataset_eval.paths_images

   print(f'Iterating eval data')
   for idx, (images, targets) in enumerate(iter(data_loader_eval)):
      # print(f'Number of images per iteration {len(images)}')
      # print(f'Loaded image: {list_images[idx]}')
      sample = fo.Sample(list_images[idx])
      detections = []
      target_boxes, target_labels = targets[0]['boxes'], targets[0]['labels']
      for box, label in zip(target_boxes, target_labels):
         # [xmin, ymin, xmax, ymax] in Pytorch Dataset | [xmin, ymin, w, h] for FO dataset
         bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
         detections.append(fo.Detection(label=CONSTANTS.NUM_TO_CATEGORIES[label.item()], bounding_box=bbox))

      sample["ground_truth"] = fo.Detections(detections=detections)
      dataset_eval_fo.add_sample(sample)


   # Training loop
   for epoch in range(num_epochs):
      # Training
      train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=2)
      # Optimizer step
      lr_scheduler.step()
      print(f'EVALUATION')

      # Evaluation
      coco_eval = evaluate(model, data_loader_eval_coco, device=device) 

      # Obtain predictions
      evaluation.evaluate(model, epoch, device, dataset_eval_fo, resized_shape)
      
      # COCO AP metrics
      map, map_staff, map_lyrics = evaluation.evaluate_coco_AP(dataset_eval_fo)
      # VOC Challenge metrics
      evaluation.evaluate_VOC(dataset_eval_fo)
      # COCO strict metrics (0.75 iou)
      evaluation.evaluate_coco_strict(dataset_eval_fo)

      # print(f'Info about eval {dataset_eval_fo.get_evaluation_info("eval")}')
      if earlyStopping.update(map, epoch) is True:
         break

      print()
   
   print(f'ENDED TRAINING')
   print()
