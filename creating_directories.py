import os
from config import (MODELS_DATA, SAVE_MODELS, TENSORBOARD_LOGS, SAVE_CURRENT_MODEL, LOGS_DIR_CURRENT_MODEL,
                    SAVE_CURRENT_TENSORBOARD_LOGS)


def create_dirs():
    os.makedirs(TENSORBOARD_LOGS, exist_ok=True)
    os.makedirs(MODELS_DATA, exist_ok=True)
    os.makedirs(SAVE_MODELS, exist_ok=True)
    os.makedirs(SAVE_CURRENT_MODEL, exist_ok=True)
    os.makedirs(LOGS_DIR_CURRENT_MODEL, exist_ok=True)
    os.makedirs(SAVE_CURRENT_TENSORBOARD_LOGS, exist_ok=True)


if __name__ == '__main__':
    create_dirs()
