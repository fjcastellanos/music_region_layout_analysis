import json
import io
import numpy as np
from PIL import Image
# from tensorflow.core.example.feature_pb2 import BytesList
import constants as CONSTANTS
# import tensorflow as tf
# from object_detection.utils import dataset_util
# import tf

def calculate_indexes(fold_validation, fold_test, n_folds=5):
   """Calculates indexes for training, validation and test. 

   Args: 
      fold_validation: int
         number of the validation fold
      fold_test: int
         number of the test fold
      n_folds: int
         number of total folds

   Returns:
      folder_idx: np.array[5]
         array with folds for each split
   """
   folder_idx = np.zeros(shape=(n_folds,), dtype=np.uint8)
   folder_idx[-1] = fold_test
   folder_idx[-2] = fold_validation
  
   for i in range(3):
      folder_idx[i] = (fold_test + i + 1) % n_folds

   return folder_idx


def convert_json_to_data(filename, path_image, resize_shape):
   """Reads a JSON and an image and returns a TF.record(Example)

   Args:
      filename: str
         filename of the json {name.jpg.json} 
      path image: str
         filename of the image {name.jpg}
      resize_shape: pair (x, y)
         contains the shape of the resized image
      

   Returns:
      image: list of [1, height, width, 3] Tensor of tf.float32
         
   """

   # print(f'Path_image: {path_image}')
   # print(f'Path_json: {filename}')
   
   image = Image.open(path_image)#.convert("RGB")
   width, height = image.size
   if resize_shape is not None:
      image = image.resize(resize_shape)
      ratio_width = width / resize_shape[0]
      ratio_height = height / resize_shape[1]

   boxes = []
   classes = []

   with open(filename) as f:
      # Read image from json (json info and image)
      example_dict = json.load(f)
      filename_image = example_dict['filename'].encode('utf8')
      if "pages" not in example_dict:
         return image, boxes, classes

      pages = example_dict['pages']
      
      for page_idx, page in enumerate(pages):
         if "regions" not in page:
            continue
         regions = page['regions']

         for box in regions: 
            box_data = box['bounding_box']
            category = box['type']
            if "staff" in category: # empty-staff => staff
               category = "staff"

            if category in CONSTANTS.CATEGORIES:
               # print(f'region: {box}')
               category_int = CONSTANTS.CATEGORIES_TO_NUM[category]
               xmin, xmax = box_data['fromX'], box_data['toX']
               ymin, ymax = box_data['fromY'], box_data['toY']
               
               if resize_shape is not None:
                  xmin, xmax = round(xmin/ratio_width, 4), round(xmax/ratio_width, 4)
                  ymin, ymax = round(ymin/ratio_height, 4), round(ymax/ratio_height, 4)

               boxes_images = [xmin, ymin, xmax, ymax]
               
               if xmax <= xmin or ymax <= ymin:
                  continue

               boxes.append(boxes_images)
               classes.append(category_int)

   return image, boxes, classes

































# def create_tf_example(example):
#   # TODO(user): Populate the following variables from your example.
#   height = None # Image height
#   width = None # Image width
#   filename = None # Filename of the image. Empty if image is not from file
#   encoded_image_data = None # Encoded image bytes
#   image_format = None # b'jpeg' or b'png'

#   xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
#   xmaxs = [] # List of normalized right x coordinates in bounding box
#              # (1 per box)
#   ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
#   ymaxs = [] # List of normalized bottom y coordinates in bounding box
#              # (1 per box)
#   classes_text = [] # List of string class name of bounding box (1 per box)
#   classes = [] # List of integer class id of bounding box (1 per box)

#   tf_example = tf.train.Example(features=tf.train.Features(feature={
#       'image/height': dataset_util.int64_feature(height),
#       'image/width': dataset_util.int64_feature(width),
#       'image/filename': dataset_util.bytes_feature(filename),
#       'image/source_id': dataset_util.bytes_feature(filename),
#       'image/encoded': dataset_util.bytes_feature(encoded_image_data),
#       'image/format': dataset_util.bytes_feature(image_format),
#       'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#       'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#       'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#       'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#       'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#       'image/object/class/label': dataset_util.int64_list_feature(classes),
#   }))
#   return tf_example