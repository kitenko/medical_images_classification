import os

DATASET_PATH = 'data'
DATASET_PATH_IMAGES = os.path.join(DATASET_PATH, 'images')
JSON_FILE_PATH = os.path.join(DATASET_PATH, 'data.json')

BATCH_SIZE = 32
NUMBER_OF_CLASSES = 9
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 0.0001
EPOCHS = 150
WEIGHTS = None

NAME_MODEL = 'EfficientNetB1'

TENSORBOARD_LOGS = 'tensorboard_logs'
LOGS_DIR = 'logs_dir'
SAVE_MODELS = 'save_models'

if WEIGHTS == None:
    LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS_DIR, NAME_MODEL + '_no_weights')
    SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, NAME_MODEL + '_no_weights')
else:
    LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS_DIR, NAME_MODEL + WEIGHTS)
    SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, NAME_MODEL + WEIGHTS)
