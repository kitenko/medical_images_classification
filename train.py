import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

from data_generator import DataGenerator
from metrics import Recall, Precision, F1Score
from config import NUMBER_OF_CLASSES, LOGS_DIR, EPOCHS, JSON_FILE_PATH
from classification_model_git import CustomModelGit
from logcallback import LogCallback

# run gpu
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.005
    if epoch > 10:
        learning_rate = 0.001
    if epoch > 20:
        learning_rate = 0.0001
    if epoch > 40:
        learning_rate = 0.00001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def train(dataset_path_json: str, save_path: str) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    :param save_path: path to save weights and training logs.
    """
    log_dir = os.path.join(save_path)
    os.makedirs(log_dir, exist_ok=True)

    train_data_gen = DataGenerator(dataset_path_json, is_train=True)
    test_data_gen = DataGenerator(dataset_path_json, is_train=False)

    model = CustomModelGit().build()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy', Recall(NUMBER_OF_CLASSES), Precision(NUMBER_OF_CLASSES),
                           F1Score(NUMBER_OF_CLASSES)])
    model.summary()

    early = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(log_dir, 'model.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                    filepath=checkpoint_filepath,
                                                                    monitor='val_accuracy',
                                                                    mode='max',
                                                                    save_best_only=True
                                                                   )
    with LogCallback() as call_back:
        model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                            validation_steps=len(test_data_gen), epochs=EPOCHS,
                            callbacks=[call_back, early], workers=8)


if __name__ == '__main__':
    train(JSON_FILE_PATH, LOGS_DIR)
