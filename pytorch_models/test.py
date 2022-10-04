from PIL.Image import CONTAINER
import torch
import numpy
import os
import utils
import evaluation
import sys
import constants as CONSTANTS
import json
from args import args_parser_test
from models import MODELS_IMAGE_SIZE, EarlyStopping, ModelCheckpoint
from train_torch import get_model_filepath
from dataset import MusicDataset, get_transform
# import fiftyone as fo
from engine import train_one_epoch, evaluate
from dataclasses import dataclass, field
from typing import List

@dataclass
class Sample:
   filepath: str
   labels: List
   bboxes: List
   json_gt: str
   pred_path: str
   gt_path: str

def get_daug_SAE(args):
   
   if args.unif_rotation is None:
      aug = 'nodaug'
   else:
      aug = 'random-auto' if args.unif_rotation == 0 else 'random-auto-uniform-rotate'

   return 'Output-' + aug


def read_SAE_results(args):
   
   aug = get_daug_SAE(args)
   n_pages = args.n_pages if args.n_pages is not None else args.n_images

   if "Mus-Tradicional" in args.dataset:
      print('MUS TRADICIONAL')
      paths = CONSTANTS.C_COMBINED if "c-combined" in args.dataset else CONSTANTS.M_COMBINED
      paths = [
         CONSTANTS.DATA_RESULTS_SAE + args.split + "/" + aug + '/' + args.dataset + '/' + paths[0] + '/JSON/', 
         CONSTANTS.DATA_RESULTS_SAE + args.split + "/" + aug + '/' + args.dataset + '/' + paths[1] + '/JSON/'
      ]
      
   else:
      print('B-59-850/SEILS')
      paths = [CONSTANTS.DATA_RESULTS_SAE + args.split + "/" + aug + '/' +args.dataset + '/JSON/']

   all_scores, all_labels, all_boxes = [], [], []
   all_files_staff, all_files_lyrics = [], []
   for path in paths:
      # Read all the staff/lyrics list
      path_staff = path + 'staff' + '/' + str(n_pages) + '/'
      path_lyrics = path + 'lyrics' + '/' + str(n_pages) + '/'
      print(f'Path staff {path_staff}')
      print(f'Path lyrics {path_lyrics}')

      files_staff, files_lyrics = os.listdir(path_staff), os.listdir(path_lyrics)
      files_staff.sort()
      files_lyrics.sort()
      
      # all_files_staff.append(files_staff)
      # all_files_lyrics.append(files_lyrics)
      print(f'Files to read: {files_staff}, {files_lyrics}')
      
      for file_staff, file_lyrics in zip(files_staff, files_lyrics):
         all_files_staff.append(file_staff)
         all_files_lyrics.append(file_lyrics)

         with open(path_staff + file_staff) as f_staff, open(path_lyrics + file_lyrics) as f_lyrics:
            print(f'FILE READING STAFF AND LYRICS {path_staff+file_staff}, {path_lyrics+file_lyrics}')
            data_staff, data_lyrics = json.load(f_staff), json.load(f_lyrics)
            scores, labels, boxes = [], [], []
            # print(f'Staff: {data_staff}, Lyrics: {data_lyrics}')
            for bbox_staff in data_staff:
               boxes.append(bbox_staff)
               labels.append(CONSTANTS.CATEGORIES_TO_NUM['staff'])
               scores.append(1.)

            for bbox_lyrics in data_lyrics:
               boxes.append(bbox_lyrics)
               labels.append(CONSTANTS.CATEGORIES_TO_NUM['lyrics'])
               scores.append(1.)

            all_scores.append(scores)
            all_boxes.append(boxes)
            all_labels.append(labels)

      # print(f'Scores: {all_scores}')
      # print(f'boxes: {all_boxes}')
      # print(f'labels: {all_labels}')
      
   all_files_staff.sort()
   all_files_lyrics.sort()

   return all_scores, all_labels, all_boxes, all_files_staff, all_files_lyrics



def get_test_folders(args):
   """Calculates and returns filenames for images and its labels (bounding boxes, classes, etc)
   Args:
      args: dict
         args of the program
      folder: np.array
         contains the folder for each division [training, validation, test]

   Returns:
      test_folders: list
         list with the paths for all json files that contains labels + information
      test_folders_images: list
         list with the paths for all json files that contains images
   """

   test_folder, test_folder_images = [], [None, None, None]
   filepath_base = CONSTANTS.DATA_ROOT_FOLDER + 'Folds/' + args.dataset + '/fold'

   #test_folder = filepath_base + str(args.fold) + '/train.txt'
   test_folder = filepath_base + str(args.fold) + '/' + args.split + '.txt'

   # OJO, CAMBIADO ESTO PARA PODER PREDECIR EL END TO END EN TRAINING-VAL-TEST

   print(f'Path for evaluation {test_folder}')
   with open(test_folder, 'r') as file:
      lines = file.readlines()
      lines[-1] = lines[-1] + '\n'
      test_folder = [lines.replace('\n', '') for lines in lines]
      lines_images = [line.replace('JSON/','').replace('datasets', 'data/SRC')[:-6] for line in lines]
      test_folder_images = lines_images
      print(f'Test folder {test_folder}')
      print(f'Test folder {test_folder_images}')
      if not os.path.isfile(test_folder_images[-1]):
         test_folder_images[-1] = test_folder_images[-1][:-1]

   return test_folder, test_folder_images


if __name__ == "__main__":
   args = args_parser_test.parse_args()
   print(args)

   resized_shape = MODELS_IMAGE_SIZE[args.model]
   print(f'FOLD: {args.fold}')
   print(f'CURRENT PATH: {os.getcwd()}')
   print(f'ARGS: {args}')
   print(f'MODEL SELECTED: {args.model}')
   print(f'SIZE OF IMAGES: {resized_shape}')
   # Gets data for test 
   test_folders, test_folders_images = get_test_folders(args)
   test_folders.sort()
   test_folders_images.sort()
   print(f'Test folder sorted: {test_folders}, {test_folders_images}')

   
   
   path_checkpoint = get_model_filepath(args)
   print(f'Path for model checkpoint: {path_checkpoint}')
   path_json_pred = path_checkpoint
   path_json_pred = path_json_pred.replace(CONSTANTS.CHECKPOINT_ROOT, CONSTANTS.PREDICTION_FOLDER)
   # OJO, CAMBIADO PARA LA RUTA TRAIN, VAL TEST DEL END-TO-END

   _n_pages = 0
   if args.n_pages is not None and args.n_images is not None:
      _n_pages = args.n_pages
   else:
      _n_pages = args.n_images

   path_json_pred = path_json_pred[:path_json_pred.rfind("/")+1] + str(_n_pages) + "_pages/" + args.split + '/'
   os.makedirs(path_json_pred, exist_ok=True) 
   print(f'Path for model prediction to end to end (json): {path_json_pred}')

   if args.model != "SAE":
      # Load the model
      #model = torch.load(path_checkpoint)
      # IN MAC FOR INFERENCE
      model = torch.load(path_checkpoint)#, map_location=torch.device('gpu'))
      print(f'MODEL LOADED')

   device = torch.device('cpu')

   # Test  dataset (no redimension in order to calculate metrics)
   dataset_eval = MusicDataset(args.dataset, test_folders, test_folders_images, None, get_transform())
   data_loader_eval = torch.utils.data.DataLoader(
      dataset_eval, batch_size=1, shuffle=False, num_workers=0, #num_workers=2, COMMENTED TO BE LAUNCHED IN MAC (LOCALLY) 
      collate_fn=utils.collate_fn
   )

   # Resized for coco (to check if metrics change)
   dataset_eval_coco = MusicDataset(args.dataset, test_folders, test_folders_images, resized_shape, get_transform())
   data_loader_eval_coco = torch.utils.data.DataLoader(
      dataset_eval_coco, batch_size=1, shuffle=False, num_workers=0, #num_workers=2, COMMENTED TO BE LAUNCHED IN MAC (LOCALLY)
      collate_fn=utils.collate_fn
   )

   print(f'STARTING TEST')
   
   print(f'Created dataset')
   list_images = dataset_eval.paths_images
   list_jsons = dataset_eval.paths_json


   print(f'Iterating eval data')
   samples = []
   for idx, (images, targets) in enumerate(iter(data_loader_eval)):
      
      _labels, _bboxes = [], []
      target_boxes, target_labels = targets[0]['boxes'], targets[0]['labels']
      for box, label in zip(target_boxes, target_labels):
         bbox = box.tolist()# [box[0], box[1], box[2], box[3]]
         _labels.append(CONSTANTS.NUM_TO_CATEGORIES[label.item()])
         _bboxes.append(bbox)
      

      json_filename = list_jsons[idx][list_jsons[idx].rfind("/")+1:]

      _pred_path = path_json_pred + "pred-" + json_filename
      _gt_path = path_json_pred + "gt-" + json_filename
      sample = Sample(list_images[idx], _labels, _bboxes, 
                      list_jsons[idx], _pred_path, _gt_path)
      samples.append(sample)

   # Evaluation (duplicated removing FO)
   if args.model != "SAE":
      evaluation.evaluate(model, 0, device, samples, resized_shape)
   else:
      scores, labels, boxes, files_staff, files_lyrics = read_SAE_results(args)
      evaluation.evaluate_SAE(device, samples, scores, labels, boxes, files_staff, files_lyrics)
   

   # # Evaluation
   # # Commented for End-to-End
   # # coco_eval = evaluate(model, data_loader_eval_coco, device) 
   # # Obtain predictions
   # if args.model != "SAE":
   #    evaluation.evaluate(model, 0, device, dataset_eval_fo, resized_shape)
   # else:
   #    scores, labels, boxes, files_staff, files_lyrics = read_SAE_results(args)
   #    evaluation.evaluate_SAE(device, dataset_eval_fo, scores, labels, boxes, files_staff, files_lyrics)
   
   
   # # Commented for End-to-End
   # # COCO AP metrics
   # map, map_staff, map_lyrics = evaluation.evaluate_coco_AP(dataset_eval_fo)
   # # VOC Challenge metrics
   # evaluation.evaluate_VOC(dataset_eval_fo)
   # # COCO strict metrics (0.75 iou)
   # evaluation.evaluate_coco_strict(dataset_eval_fo)
   
   print(f'ENDED TEST')

   