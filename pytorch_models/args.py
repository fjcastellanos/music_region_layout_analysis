import argparse
from models import MODELS
import constants as CONSTANTS

# Argument parser for training and validation
args_parser_training = argparse.ArgumentParser()

args_parser_training.add_argument("model", help="model to train", choices=MODELS, type=str)
args_parser_training.add_argument("dataset", help="dataset for training and validation", 
                                  choices=CONSTANTS.DATASETS, type=str)

args_parser_training.add_argument("--n_pages", help="number of pages for generation", type=int)
args_parser_training.add_argument("--n_images", help="number of images for training", type=int)
args_parser_training.add_argument("--unif_rotation", help="type of rotation", type=int)
args_parser_training.add_argument("--batch_size", help="batch size for training", type=int)
args_parser_training.add_argument("fold", help="fold number for training, validation and test", type=int, choices=[0,1,2,3,4])
args_parser_training.add_argument("epochs", help="epochs for training", type=int)
args_parser_training.add_argument("patience", help="patience for training", type=int)

# Argument parser for training
args_parser_test = argparse.ArgumentParser()
args_parser_test.add_argument("model", help="model to train", choices=MODELS, type=str)
args_parser_test.add_argument("dataset", help="dataset for training and validation", choices=CONSTANTS.DATASETS, type=str)
args_parser_test.add_argument("--n_pages", help="number of pages for generation", type=int)
args_parser_test.add_argument("--n_images", help="number of images for training", type=int)
args_parser_test.add_argument("--unif_rotation", help="type of rotation", type=int)
args_parser_test.add_argument("--split", help="training, validation or test split to do infernce", type=str, choices=['train', 'val', 'test'])
args_parser_test.add_argument("fold", help="fold number for training, validation and test", type=int, choices=[0,1,2,3,4])

