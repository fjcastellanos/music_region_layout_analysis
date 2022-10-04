# Constants file

# Training Constants
N_FOLDS = 5 
TRAIN_FILENAME = 'train.txt'
VALIDATION_FILENAME = 'val.txt'
CHECKPOINT_ROOT = 'checkpoint/'
CHECKPOINT_EXTENSION = '.pt'

# Folders paths
SUFIX_FOLDS = "/Folds/"
FOLDER_DAUG = 'daug100/'
PREFIX_RND_AUTO = 'random-auto/'
PREFIX_RND_AUTO_UNIF = 'random-auto-uniform-rotate/'
DATA_ROOT_FOLDER = 'data/'
RESULTS_SAE = 'results_SAE/'
DATA_RESULTS_SAE = DATA_ROOT_FOLDER + RESULTS_SAE



# Precitions and testing paths
# WARNING: changed for end_to_end data directly 
END_TO_END_FOLDER = 'end_to_end/data/'
PREDICTION_FOLDER = END_TO_END_FOLDER + 'outputs_json/'

# Thresold for matching to end-to-end
MATCHING_THRESHOLD = 0.55


# Categories
CATEGORIES = ['staff', 'lyrics']
CATEGORIES_TO_NUM = dict({
  'background': 0,
  'staff': 1,
  'empty-staff': 1,
  'lyrics': 2,
})
NUM_TO_CATEGORIES = dict({
  0: 'background',
  1: 'staff',
  2: 'lyrics' 
}) 

NUM_CLASSES = 2

# Datasets
DATASETS = [
  'Mus-Tradicional/c-combined',
  'Mus-Tradicional/m-combined',
  'Mus-Tradicional/mision02',
  'SEILS', 
  'b-59-850'
]

C_COMBINED = ['c5', 'c14']
M_COMBINED = ['m16','m38']



