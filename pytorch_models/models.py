# Container for all models we want to test

import torch
import torchvision
import numpy as np
import constants as CONSTANTS
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

#Faster R-CNN w. ResNet50 backbone
def faster_rcnn():
  print(f'Selecting Faster R-CNN with resnet-50')
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  num_classes = CONSTANTS.NUM_CLASSES + 1 # Staff, Lyrics + Background
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  
  return model

def mask_rcnn_resnet50():
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
  num_classes = CONSTANTS.NUM_CLASSES + 1 # Staff, Lyrics + Background
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model

# RetinaNet w. ResNet50 Backbone
def RetinaNet():
  print(f'Loading RetinaNet con ResNet50')
  model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, progress=True)

  return model

# Faster R-CNN w. MobileNet backbone
def MobileNet():
  print(f'Selecting Faster R-CNN with MobileNet v3')
  model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
  num_classes = CONSTANTS.NUM_CLASSES + 1 # Staff, Lyrics + Background
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model

# SSD w. VGG16 backbone
def ssd_vgg16():
  model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

  return model

def YOLO():
  return None

def get_model(args):
  return None
  

MODELS = ['Faster-RCNN', 'RetinaNet', 'MobileNet', 'YOLO', 'Mask-RCNN', 'SSD-VGG16', 'SAE']

MODELS_FUNC = dict({
  'Faster-RCNN':faster_rcnn, 
  'RetinaNet':RetinaNet, 
  'MobileNet':MobileNet,
  'YOLO':None,
  'Mask-RCNN':None,
  'SSD-VGG16':ssd_vgg16
})

MODELS_IMAGE_SIZE = dict(
  {'Faster-RCNN': (512, 512),
  'RetinaNet': (512, 512),
  'MobileNet': (512, 512),
  'YOLO': (512, 512),
  'Mask-RCNN': (512, 512), 
  'SSD-VGG16': (512, 512),
  'SAE': None
})





class ModelCheckpoint:
  def __init__(self, filepath, model):
    """
      Initialises the class for a ModelCheckpoint callback.
      Arguments: 
        filepath: string
          filename with .pt extension to save the model
        model: torch.nn
          model to save while training
    """
    self.filepath = filepath
    self.model = model
    self.save()
  
  def save(self):
    torch.save(self.model, self.filepath)

      
class EarlyStopping:
  def __init__(self, patience, epochs, checkpoint, epsilon):
    """
      Initialises the class for an Early Stopping callback.
      Arguments: 
        patience: int
          number of epochs without improving the metric
        epochs: int
          total number of epochs
        checkpoint: ModelCheckpoint
          object for saving the best model
        epsilon: float
          threshold in order to update best result
    """
    self.patience = patience
    self.epochs = epochs
    self.checkpoint = checkpoint
    self.stop_training = False
    self.actual_patience = 1
    self.best_result = 0.0
    self.epsilon = epsilon

  def update(self, last_result, epoch):
    """
      Updates results after each epoch and saves the best model
      through ModelCheckpoint Object.
      Arguments:
        last_result: float
          result obtained on the selected metric
      Returns:
        Boolean
          True if the current patience has achieved maximum patience
          False otherwise
    """
    last_result = np.round(last_result, 3)
    if last_result - self.best_result < self.epsilon:
      self.actual_patience += 1
      if self.actual_patience >= self.patience:
        self.stop_training = True
        print(f'Training stopped with patience {self.patience}')
        print(f'Best result obtained: {np.round(self.best_result, 3)}')
      print(f'Not updating best result ({self.best_result}), last: {last_result}.') 
    else:
      self.actual_patience = 1
      last_best = self.best_result
      self.best_result = last_result
      # Saving best model
      self.checkpoint.save()
      print(f'Updating best result ({self.best_result}), last: {last_best}. Difference of {np.round(abs(last_best-self.best_result), 3)}.') 
      print(f'Saving on {self.checkpoint.filepath}')
    
    print(f'Early stopping: [{self.actual_patience}/{self.patience}] on epoch {epoch}/{self.epochs}')

    return self.stop_training
