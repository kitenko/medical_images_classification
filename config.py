import os

DATASET_PATH = 'data'
DATA_PATH_cecum = os.path.join(DATASET_PATH, 'cecum')
DATASET_PATH_dyed_lifted_polyps = os.path.join(DATASET_PATH, 'dyed-lifted-polyps')
DATASET_PATH_dyed_resection_margins = os.path.join(DATASET_PATH, 'dyed-resection-margins')
DATASET_PATH_retroflex_stomach = os.path.join(DATASET_PATH, 'retroflex-stomach')
DATASET_PATH_bbps_2_3 = os.path.join(DATASET_PATH, 'bbps-2-3')
DATASET_PATH_polyps = os.path.join(DATASET_PATH, 'polyps')
DATASET_PATH_z_line = os.path.join(DATASET_PATH, 'z-line')
DATASET_PATH_bbps_0_1 = os.path.join(DATASET_PATH, 'bbps-0-1')
DATASET_PATH_pylorus = os.path.join(DATASET_PATH, 'pylorus')
JSON_FILE_PATH = os.path.join(DATASET_PATH, 'data.json')
# create logs_dir
MAKE_LOGS_DIR = os.makedirs('logs_dir/resnet50_weights_imagenet', exist_ok=True)
LOGS_DIR = 'logs_dir/resnet50_weights_imagenet'
MAKE_DIR_FOR_MODEL_SAVE = os.makedirs('save_models/resnet50_weights_imagenet', exist_ok=True)
SAVE_MODELS = 'save_models/resnet50_weights_imagenet'

CLASS_NAMES = ('cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'retroflex-stomach', 'bbps-2-3', 'polyps',
               'z-line', 'bbps-0-1', 'pylorus')
BATCH_SIZE = 8
NUMBER_OF_CLASSES = 9
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 0.0001
EPOCHS = 150
WEIGHTS = 'imagenet'

NAME_MODEL = 'resnet50'
