import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from data_generator import DataGenerator
from metrics import Recall, Precision, F1Score
from classification_model_git import build_model
from logcallback import LogCallback
from config import (NUMBER_OF_CLASSES, EPOCHS, JSON_FILE_PATH, LEARNING_RATE, MODEL_NAME, SAVE_CURRENT_MODEL,
                    SAVE_CURRENT_TENSORBOARD_LOGS, TENSORBOARD_LOGS, MODELS_DATA, SAVE_MODELS, LOGS_DIR_CURRENT_MODEL)


def train(dataset_path_json: str) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    """
    # creating directories
    for p in [TENSORBOARD_LOGS, MODELS_DATA, SAVE_MODELS, SAVE_CURRENT_MODEL, LOGS_DIR_CURRENT_MODEL,
              SAVE_CURRENT_TENSORBOARD_LOGS]:
        os.makedirs(p, exist_ok=True)

    train_data_gen = DataGenerator(dataset_path_json, is_train=True)
    test_data_gen = DataGenerator(dataset_path_json, is_train=False)

    model = build_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy', Recall(NUMBER_OF_CLASSES), Precision(NUMBER_OF_CLASSES),
                           F1Score(NUMBER_OF_CLASSES)])
    model.summary()

    early = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(SAVE_CURRENT_MODEL, MODEL_NAME + '.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='F1_score',
        mode='max',
        save_best_only=True
    )
    tensor_board = tf.keras.callbacks.TensorBoard(SAVE_CURRENT_TENSORBOARD_LOGS, update_freq='batch')
    with LogCallback() as call_back:
        model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                            validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8,
                            callbacks=[call_back, early, model_checkpoint_callback, tensor_board])


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    train(JSON_FILE_PATH)
