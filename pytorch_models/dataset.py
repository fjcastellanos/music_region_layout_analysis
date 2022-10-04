import os
import numpy as np
import torch
import torchvision
from PIL import Image 
from utilss import convert_json_to_data
import transforms as T
# from torchvision.transforms import transforms as T


def filter_dataset(paths_json, paths_images, resize_shape):
  filtered_jsons, filtered_images = [], []
  for path_json, path_image in zip(paths_json, paths_images):
    img, boxes, labels = convert_json_to_data(path_json, path_image, resize_shape)
    if len(labels) > 0:
      filtered_jsons.append(path_json)
      filtered_images.append(path_image)
    else:
      print(f'FILTERED IMAGE ON DATASET (PATH): {path_json}')

  return filtered_jsons, filtered_images


class MusicDataset(torch.utils.data.Dataset):
  def __init__(self, name, paths_json, paths_images, resize_shape, transforms):
    """
      Arguments:
        name: Name of the dataset (b-59-80, Seils, Mus-Trad-...)
        path_json: list with all paths to json
        path_json: list with all paths to source images
    """
    self.name = name
    self.resize_shape = resize_shape
    self.paths_json = sorted(paths_json)
    self.paths_images = sorted(paths_images)
    self.transforms = transforms


  def __getitem__(self, idx):
    path_img = self.paths_images[idx]
    path_json = self.paths_json[idx]

    if self.resize_shape is not None: 
      img, boxes, labels = convert_json_to_data(path_json, path_img, self.resize_shape)
    else:
      img, boxes, labels = convert_json_to_data(path_json, path_img, None)

    if len(labels) <= 0: # If image does not contain any object (background)
      boxes = torch.zeros((0, 4), dtype=torch.float32)
      labels = torch.zeros(1, dtype=torch.int64)
    else:
      boxes = torch.as_tensor(boxes, dtype=torch.float32) # [xmin, ymin, xmax, ymax]
      labels = torch.as_tensor(labels, dtype=torch.int64)

    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    if self.transforms is not None:
        #img, target = self.transforms(img, target)
        img = self.transforms(img)

    return img, target

  def __len__(self):
    return len(self.paths_images)



def get_transform():
  transforms = []
  transforms.append(T.ToTensor())
  # transforms.append(torchvision.transforms.Normalize(
  #       mean=[0.485, 0.456, 0.406],
  #       std=[0.229, 0.224, 0.225],
  #   ))

  return T.Compose(transforms)