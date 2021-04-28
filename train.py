import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import config
from data_generator import DataGenerator
# from custom_model import ClassificationPathologies
from resnet_github_tensorflow.resnet import resnet_18, resnet_34, resnet_101, resnet_152
# from resnet_model import CustomResNetModel
from metrics import Recall, Precision, F1Score
from config import (NUMBER_OF_CLASSES, LOGS_DIR, INPUT_SHAPE, EPOCHS, JSON_FILE_PATH,
                    PATH_FOR_SAVE_LOGS_SCALAR)

# run gpu
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.01
    if epoch > 10:
        learning_rate = 0.001
    if epoch > 20:
        learning_rate = 0.0001
    if epoch > 50:
        learning_rate = 0.00001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def get_model():
    """
    This function returns model.

    :return: return model
    """
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()

    model.build(input_shape=INPUT_SHAPE)
    model.summary()
    return model


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

    model = get_model()
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
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=PATH_FOR_SAVE_LOGS_SCALAR,
                                                histogram_freq=0,
                                                write_graph=False,
                                                write_images=False,
                                                update_freq='epoch',
                                                profile_batch=2,
                                                embeddings_freq=1)]

    model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                        validation_steps=len(test_data_gen), epochs=EPOCHS,
                        callbacks=[model_checkpoint_callback, early, callbacks], workers=8)

    #model.save('resnet_model_git', save_format='tf')


if __name__ == '__main__':
    train(JSON_FILE_PATH, LOGS_DIR)

