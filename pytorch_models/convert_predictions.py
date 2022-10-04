# Convert predictions script in order to parse predictions to MURET format for end-to-end training
import numpy as np
import constants as CONSTANTS
import json
import copy

def calculate_iou(pred_box, gt_box):
    """Calculates Intersection Over Union (iou) over two given boxes.
      Inputs:
        pred_box, gt_box: [float] (x1, y1, x2, y2 format)
      Output:
        Intersection Over Union (IoU) calculated -> float
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box

    xi1 = max(pred_x1, gt_x1)
    yi1 = max(pred_y1, gt_y1)
    xi2 = min(pred_x2, gt_x2)
    yi2 = min(pred_y2, gt_y2)

    intersection = max(xi2 - xi1 + 1, 0) * max(yi2 - yi1 + 1, 0) # if negative, there is no intersection

    area_pred = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    area_gt = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
    union = area_pred + area_gt

    iou = intersection / float((union - intersection))
    
    return iou

def get_bounding_box(dict_bbox):
  return [dict_bbox['fromX'], dict_bbox['fromY'], dict_bbox['toX'], dict_bbox['toY']]


def convert_bbox_to_format(bbox):
  # x1, y1, x2, y2 ==> to x1, x2, y1, y2
  return dict({'fromX': int(bbox[0]), 'toX': int(bbox[2]), 'fromY': int(bbox[1]), 'toY': int(bbox[3])})

def write_predictions(gt_predictions, gt_filepath, prediction, output_pred_filepath, output_gt_filepath):
  """ Writes predictions on the filepaths. 
      Args:
        gt_predictions: groundtruth containing Tuple([bounding_boxes], [labels], [scores]) in fiftyone format (Detection class)
        gt_filepath: (string) filepath with .json format that contains the ground truth
        prediction: Tuple([bounding_boxes], [labels], [scores])
        output_pred_filepath: (string) filepath with .json format to write the output with the matches
        output_gt_filepath: (string) filepath with .json format to write the output with the matchings (original gt)
  """ 
 
  bboxes, labels, scores = prediction[0], prediction[1], prediction[2]
  
  pred_dict = dict({"bboxes": bboxes, "classes": labels, "scores": scores})

  
  gt_dict = dict({"bboxes": gt_predictions.bboxes, "classes": gt_predictions.labels})

  with open(output_pred_filepath, "w") as out_pred_file:
    out_pred_file.write(json.dumps(pred_dict))
        
  with open(output_gt_filepath, "w") as out_gt_file:
    out_gt_file.write(json.dumps(gt_dict))



def calculate_matching(gt_filepath, prediction, output_pred_filepath, output_gt_filepath):
  """ Calculates the match between GT and predictions for a given image.
      Args:
        gt_filepath: (string) filepath with .json format that contains the ground truth
        prediction: Tuple([bounding_boxes], [labels], [scores])
        output_pred_filepath: (string) filepath with .json format to write the output with the matches
        output_gt_filepath: (string) filepath with .json format to write the output with the matchings (original gt)
  """
  print(f'Calculate matching output_pred (pred) {output_pred_filepath}')
  print(f'Calculate matching output_pred (GT)   {output_gt_filepath}')

  with open(gt_filepath) as f:
      # Read image from json (json info and image)
      gt_dict = json.load(f) # Original ground truth
      pred_dict = copy.deepcopy(gt_dict)
      selected_gt_dict = copy.deepcopy(gt_dict)
      # gt_dict['pages'][0]['regions'][X]
      # gt_dict['pages'][n_page]['regions'][n_object]

      # Delete pred and selected
      if "pages" not in pred_dict:
        print(f'EMPTY JSON. key "PAGES" not found for filepath {gt_filepath}')
      
      else:
        pages = pred_dict['pages']
        for page_idx, page in enumerate(pages):
          if "regions" not in page:
              continue
          regions = page['regions']
          regions.clear()

        pages = selected_gt_dict['pages']
        for page_idx, page in enumerate(pages):
          if "regions" not in page:
              continue
          regions = page['regions']
          regions.clear()

        bboxes, labels, scores = prediction[0], prediction[1], prediction[2]
        for bbox_pred, label_pred, score_pred in zip(bboxes, labels, scores):
          best_iou, best_score, best_page_idx, best_obj_idx, best_object = 0.0, 0.0, -1, -1, None
          # (x1, x2, y1, y2 format for MURET datasets)
          # (x1, y1, x2, y2 format)
          pages = gt_dict['pages']
          is_selected = False
          for page_idx, page in enumerate(pages):
            if "regions" not in page:
              continue
            regions = page['regions']
            
            # Select the match with the best iou
            for object_idx, object in enumerate(regions):
              object_gt, bbox_gt, type_gt = object, get_bounding_box(object['bounding_box']), object['type']
              n_page, n_object = page_idx, object_idx
              # print(f'Label pred {label_pred}, gt label: {type_gt}')
              # print(f'Same label... {type_gt == label_pred}')
              if type_gt == label_pred:
                iou = calculate_iou(bbox_pred, bbox_gt)
                iou = iou if iou > CONSTANTS.MATCHING_THRESHOLD else 0.0
                # print(f'iou calculated: {iou}')
                if iou > best_iou:
                  is_selected = True
                  # print(f'Matched with iou: {iou}')
                  best_page_idx, best_obj_idx, best_object = n_page, n_object, object_gt
                  best_iou, best_score = iou, float(score_pred)

            # len_page = len(gt_dict['pages'][page_idx]['regions'])
            # print(f'Len of GT page {page_idx}: {len_page}')

          if is_selected is True: 
            selected_object = gt_dict['pages'][best_page_idx]['regions'].pop(best_obj_idx)
            selected_gt_dict['pages'][best_page_idx]['regions'].append(selected_object)
            pred_object = copy.deepcopy(best_object)
            # print(f'Pred (object gt) object: {best_object}')
            pred_object['bounding_box'] = convert_bbox_to_format(bbox_pred)
            pred_object['iou_selected'] = best_iou
            pred_object['score'] = best_score
            pred_dict['pages'][best_page_idx]['regions'].append(pred_object)

        # print(f'Predicted dictionary')
        # print(f'{pred_dict}')
        # print(f'GT selected dictionary')
        # print(f'{selected_gt_dict}')

        
        with open(output_pred_filepath, "w") as out_pred_file:
          out_pred_file.write(json.dumps(pred_dict))
        
        with open(output_gt_filepath, "w") as out_gt_file:
          out_gt_file.write(json.dumps(selected_gt_dict))

