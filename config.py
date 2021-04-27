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
CLASS_NAMES = ('cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'retroflex-stomach', 'bbps-2-3', 'polyps',
               'z-line', 'bbps-0-1', 'pylorus')
BATCH_SIZE = 8
NUMBER_OF_CLASSES = 9
INPUT_SHAPE = (None, 224, 224, 3)
LEARNING_RATE = 0.0001
EPOCHS = 5
LOGS_DIR = os.path.join(DATASET_PATH, 'logs_dir')

# choose a network
# model = "resnet18"
model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"

