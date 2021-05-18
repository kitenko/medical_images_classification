import os
from datetime import datetime

DATASET_PATH = 'data'
DATASET_PATH_IMAGES = os.path.join(DATASET_PATH, 'images')
JSON_FILE_PATH_DATA_GEN = os.path.join(DATASET_PATH, 'data.json')
JSON_FILE_PATH_INDEX_CLASS = os.path.join(DATASET_PATH, 'index_class.json')

BATCH_SIZE = 32
NUMBER_OF_CLASSES = 9
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 0.0001
EPOCHS = 150
WEIGHTS = 'imagenet'
USE_AUGMENTATION = False
SAVE_MODEL_EVERY_EPOCH = False

MODEL_NAME = 'EfficientNetB0'

MODELS_DATA = 'models_data'
TENSORBOARD_LOGS = os.path.join(MODELS_DATA, 'tensorboard_logs')
SAVE_MODELS = os.path.join(MODELS_DATA, 'save_models')
LOGS = os.path.join(MODELS_DATA, 'logs')

date_time_for_save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS, MODEL_NAME + '_' + str(WEIGHTS) + '_' + date_time_for_save + '_' +
                                      str(USE_AUGMENTATION))
SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, MODEL_NAME + '_' + str(WEIGHTS) + '_' + date_time_for_save + '_' +
                                      str(USE_AUGMENTATION))
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, MODEL_NAME + '_' + str(WEIGHTS) + '_' +
                                             date_time_for_save + '_' + str(USE_AUGMENTATION))
