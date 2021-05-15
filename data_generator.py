import json
from typing import Tuple

import cv2
from tensorflow import keras
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from config import JSON_FILE_PATH, NUMBER_OF_CLASSES, BATCH_SIZE, INPUT_SHAPE, USE_AUGMENTATION


class DataGenerator(keras.utils.Sequence):
    def __init__(self, json_path: str = JSON_FILE_PATH, batch_size: int = BATCH_SIZE, is_train: bool = True,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES,
                 augmentation_data: bool = USE_AUGMENTATION) -> None:
        """
        Data generator for the task of colour images classifying.

        :param json_path: this is path for json file.
        :param batch_size: number of images in one batch.
        :param is_train: if is_train = True, then we work with train images, otherwise with test.
        :param image_shape: this is image shape (height, width, channels).
        :param num_classes: number of image classes.
        :param augmentation_data: if this parameter is True, then augmentation is applied to the training dataset.
        """
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.augmentation_data = augmentation_data

        # read json
        with open(json_path) as f:
            data = json.load(f)

        if is_train:
            data = data['train']
            augmentation = images_augmentation(train_data=self.augmentation_data)
        else:
            data = data['test']
            augmentation = images_augmentation(train_data=False)

        self.aug = augmentation
        self.data = list(data.items())
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of data at the end of each epoch.

        """
        np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function makes batch.

        :param batch_idx: batch number.
        :return: image tensor and list with labels tensors for each output.
        """
        batch = self.data[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        labels = np.zeros((self.batch_size, self.num_classes))
        for i, (img_path, class_name) in enumerate(batch):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.aug(image=img)
            img = img['image']
            images[i, :, :, :] = img
            labels[i, class_name['index']] = 1
        images = image_normalization(images)
        return images, labels

    def show(self, batch: Tuple[np.ndarray, np.ndarray], train_data: bool = USE_AUGMENTATION) -> None:
        """
        This method showing image with label.

        :param batch: this is input batch from __getitem__.
        :param train_data: if this parameter is True then augmentation is applied to train dataset.
        """
        images, labels = batch[0], batch[1]

        rows_columns_subplot = self.batch_size
        while np.math.sqrt(rows_columns_subplot) - int(np.math.sqrt(rows_columns_subplot)) != 0.0:
            rows_columns_subplot += 1
        rows_columns_subplot = int(np.math.sqrt(rows_columns_subplot))

        plt.figure(figsize=[20, 20])
        for i, j in enumerate(images):
            lable_image = np.where(labels[i] == 1)
            plt.subplot(rows_columns_subplot, rows_columns_subplot, i+1)
            plt.imshow(j)
            if train_data is True:
                plt.title('Augmented, class = "{}"'.format(lable_image))
            else:
                plt.title('Original, class = "{}"'.format(lable_image))
        if plt.waitforbuttonpress(0):
            plt.close('all')
            raise SystemExit
        plt.close()


def images_augmentation(train_data: bool = USE_AUGMENTATION) -> A.Compose:
    """
    This function makes augmentation data.

    :param train_data: if this parameter is True then augmentation is applied to train dataset.
    :return: augment data
    """
    if train_data is True:
        aug = A.Compose([
              A.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1]),
              A.Blur(blur_limit=(1, 4), p=0.2),
              A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.2),
              A.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.1, hue=0.0, p=0.2),
              A.Equalize(mode='cv', by_channels=True, mask=None, p=0.1),
              A.Flip(p=0.4),
              A.Rotate(limit=320, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False,
                       p=0.2)
              ])
    else:
        aug = A.Compose([A.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1])])

    return aug


def image_normalization(image: np.ndarray) -> np.ndarray:
    """
    Image normalization.
    :param image: image numpy array.
    :return: normalized image.
    """
    return image / 255.0


if __name__ == '__main__':
    x = DataGenerator()
    for i in range(len(x)):
        x.show(x[i])
