import os
from datetime import datetime

date_time_for_save = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DATASET_PATH = 'data'
DATASET_PATH_IMAGES = os.path.join(DATASET_PATH, 'images')
JSON_FILE_PATH = os.path.join(DATASET_PATH, 'data.json')

BATCH_SIZE = 16
NUMBER_OF_CLASSES = 9
INPUT_SHAPE = (240, 240, 3)
LEARNING_RATE = 0.0001
EPOCHS = 150
WEIGHTS = 'imagenet'
AUGMENTATION_DATA = True

NAME_MODEL = 'EfficientNetB1'

MODELS_DATA = 'models_data'
TENSORBOARD_LOGS = os.path.join(MODELS_DATA, 'tensorboard_logs')
SAVE_MODELS = os.path.join(MODELS_DATA, 'save_models')
LOGS = os.path.join(MODELS_DATA, 'logs')

LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS, NAME_MODEL + '_' + str(WEIGHTS) + '_' + date_time_for_save)
SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, NAME_MODEL + '_' + str(WEIGHTS) + '_' + date_time_for_save)
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, NAME_MODEL + '_' + str(WEIGHTS) + '_' +
                                             date_time_for_save)
