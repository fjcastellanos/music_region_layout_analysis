# evaluation
import torchvision
from torchvision.transforms import functional as func
import torch
# import fiftyone as fo
# import fiftyone.utils.coco as fouc
import numpy as np
from PIL import Image
import constants as CONSTANTS
import cv2 as cv
import math
import convert_predictions as conv_pred
import os



def evaluate_SAE(device, dataset_eval_fo, scores, labels, boxes, files_staff, files_lyrics):
  print("EVALUATING SAE")
  print(f'Files read for staff {files_staff}')
  print(f'Files read for lyrics {files_lyrics}')


  for idx_image, sample in enumerate(dataset_eval_fo):
      image = Image.open(sample.filepath)
      image_orig, (w, h) = image, image.size
      print(f'Image filepath: {sample.filepath}')
      print(f'Image filepath json: {sample.json_gt}')

      opencv_image = image_orig 
      opencv_image = cv.cvtColor(np.array(opencv_image), cv.COLOR_RGB2BGR)
      for box in boxes[idx_image]: #[:12]:
          bbox = box
          # x1, y1, x2, y2
          # print(f'PRINTING {bbox}')
          starting_coords = (int(bbox[1]), int(bbox[0]))
          ending_coords = (int(bbox[3]), int(bbox[2]))
          color = (255, 0, 0) # BGR color
          opencv_image = cv.rectangle(opencv_image, starting_coords, ending_coords, color, 2)
      
      image_name = sample.filepath[sample.filepath.rfind("/"):]
      image_name = image_name[:-4] # + "/"
      output_image = 'outputs/' + image_name + "-" + str(idx_image) + '.jpg'
      cv.imwrite(output_image, opencv_image)

      # Detections for End-to-End
      _labels, _scores, _boxes = [], [], []

      # Detections to FiftyOne format
      # detections = []
        
      for label, score, box in zip(labels[idx_image], scores[idx_image], boxes[idx_image]):
        x1, y1, x2, y2 = box
        # x1, x2, y1, y2 = box
        rel_box = [y1, x1, (y2 - y1), (x2 - x1)]
        _boxes.append(box)
        _label=CONSTANTS.NUM_TO_CATEGORIES[label] if label in CONSTANTS.NUM_TO_CATEGORIES else 'background' #CONSTANTS.NUM_TO_CATEGORIES[label],
        _labels.append(_label)
        _scores.append(score)

      conv_pred.write_predictions(sample, sample.json_gt, (_boxes, _labels, _scores), sample.pred_path, sample.gt_path)

  print(f'Ended detections for SAE model')




def evaluate(model, epoch, device, dataset_eval_fo, resized_shape):
  print(f'Evaluation with FO dataset')
  # Evaluation
  for idx_image, sample in enumerate(dataset_eval_fo):
      image = Image.open(sample.filepath)
      image_orig, (w, h) = image, image.size
      
      print(f'Image original size {image_orig.size}')
      print(f'Image filepath: {sample.filepath}')
      print(f'Image filepath json: {sample.json_gt}')
      print(f'Json filepath to write (pred) the matching: {sample.pred_path}')
      print(f'Json filepath to write (gt)   the matching: {sample.gt_path}')

      image = image.resize(resized_shape)
      opencv_image = image_orig if epoch % 5 == 0 else None
      image = func.to_tensor(image).to(device)
      # c, h, w = image.shape
      ratio_width = w / resized_shape[0]
      ratio_height = h / resized_shape[1]

      # Inference
      model.eval()
      preds = model([image])[0]
      labels = preds["labels"].cpu().detach().numpy()
      scores = preds["scores"].cpu().detach().numpy()
      boxes = preds["boxes"].cpu().detach().numpy()

      print(f'Number of preds: {scores.shape}')


      # Detections for End-to-End
      _labels, _scores, _boxes = [], [], []

      # Removing FO

      # Detections to FiftyOne format
      detections = []
      for label, score, box in zip(labels, scores, boxes):
        x1, y1, x2, y2 = box
        # rel_box = [x1 * ratio_width, y1 * ratio_height, (x2 - x1) * ratio_width, (y2 - y1) * ratio_height]
        
        # End-to-end format (original)
        # _boxes.append(np.round(np.array([x1 * ratio_width, y1 * ratio_height, x2 * ratio_width, y2 * ratio_height]), 1))
        _boxes.append([float(x1 * ratio_width), 
                       float(y1 * ratio_height), 
                       float(x2 * ratio_width), 
                       float(y2 * ratio_height)])
        _label=CONSTANTS.NUM_TO_CATEGORIES[label] if label in CONSTANTS.NUM_TO_CATEGORIES else 'background' #CONSTANTS.NUM_TO_CATEGORIES[label],
        _labels.append(_label)
        _scores.append(float(round(score, 2)))


      conv_pred.write_predictions(sample, sample.json_gt, (_boxes, _labels, _scores), sample.pred_path, sample.gt_path)



def evaluate_coco_AP(dataset_eval_fo):
  # COCO AP
  print(f'RESULTS FOR 0.5:0.9:0.05')
  results = dataset_eval_fo.evaluate_detections(
      "predictions", gt_field="ground_truth", eval_key="eval", compute_mAP=True,
  )
  map, map_staff, map_lyrics = np.round(np.array([results.mAP(), results.mAP(classes=["staff"]), results.mAP(classes=["lyrics"])]), 3)
  print(f'MAP CALCULATED {map}')
  print(f'MAP (STAFF) CALCULATED {map_staff}')
  print(f'MAP (LYRICS) CALCULATED {map_lyrics}')
  # print(f'MAP (BACKGROUND) CALCULATED {map_background}')

  # Get the 10 most common classes in the dataset
  counts = dataset_eval_fo.count_values("ground_truth.detections.label")
  classes = sorted(counts, key=counts.get, reverse=True)

  # Print a classification report for the top-10 classes
  results.print_report(classes=classes)
  results.metrics(classes=classes)

  return map, map_staff, map_lyrics


def evaluate_VOC(dataset_eval_fo):
  # VOC CHALLENGE
  print(f'RESULTS FOR 0.5')
  results = dataset_eval_fo.evaluate_detections(
      "predictions", gt_field="ground_truth", eval_key="eval", compute_mAP=True,
      iou=0.5,
      iou_threshs=[0.5]
  )

  map, map_staff, map_lyrics = np.round(np.array([results.mAP(), results.mAP(classes=["staff"]), results.mAP(classes=["lyrics"])]), 3)
  print(f'MAP CALCULATED {map}')
  print(f'MAP (STAFF) CALCULATED {map_staff}')
  print(f'MAP (LYRICS) CALCULATED {map_lyrics}')

    # Get the 10 most common classes in the dataset
  counts = dataset_eval_fo.count_values("ground_truth.detections.label")
  classes = sorted(counts, key=counts.get, reverse=True)

  # Print a classification report for the top-10 classes
  results.print_report(classes=classes)
  results.metrics(classes=classes)

def evaluate_coco_strict(dataset_eval_fo):
  # AP STRICT
  print(f'RESULTS FOR 0.75')
  results = dataset_eval_fo.evaluate_detections(
      "predictions", gt_field="ground_truth", eval_key="eval", compute_mAP=True,
      iou=0.75,
      iou_threshs=[0.75]
  )

  map, map_staff, map_lyrics = np.round(np.array([results.mAP(), results.mAP(classes=["staff"]), results.mAP(classes=["lyrics"])]), 3)
  print(f'MAP CALCULATED {map}')
  print(f'MAP (STAFF) CALCULATED {map_staff}')
  print(f'MAP (LYRICS) CALCULATED {map_lyrics}')

    # Get the 10 most common classes in the dataset
  counts = dataset_eval_fo.count_values("ground_truth.detections.label")
  classes = sorted(counts, key=counts.get, reverse=True)

  # Print a classification report for the top-10 classes
  results.print_report(classes=classes)
  results.metrics(classes=classes)

