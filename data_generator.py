import os
import json
from typing import Tuple

import cv2
from tensorflow import keras
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from config import JSON_FILE_PATH, NUMBER_OF_CLASSES, BATCH_SIZE, INPUT_SHAPE


class DataGenerator(keras.utils.Sequence):
    def __init__(self, json_path: str = JSON_FILE_PATH, batch_size: int = BATCH_SIZE, is_train: bool = True,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES) -> None:
        """
        Data generator for the task of colour images classifying.

        :param json_path: this is path for json file.
        :param batch_size: number of images in one batch.
        :param is_train: if is_train = True, then we work with train images, otherwise with test.
        :param image_shape: this is image shape (height, width, channels).
        :param num_classes: number of image classes.
        """
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_shape = image_shape
        self.num_classes = num_classes

        # read json
        with open(json_path) as f:
            self.data = json.load(f)

        if is_train:
            self.data = self.data['train']
            augmentations = A.Compose([
                    A.Resize(height=self.image_shape[0], width=self.image_shape[1]),
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Transpose(),
                    A.OneOf([
                        A.IAAAdditiveGaussianNoise(),
                        A.GaussNoise(),
                    ], p=0.2),
                    A.OneOf([
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                    A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=.1),
                        A.IAAPiecewiseAffine(p=0.3),
                    ], p=0.2),
                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.IAASharpen(),
                        A.IAAEmboss(),
                        A.RandomBrightnessContrast(),
                    ], p=0.3),
                    A.HueSaturationValue(p=0.3)
                    ])
        else:
            self.data = self.data['test']
            augmentations = A.Compose([A.Resize(height=self.image_shape[0], width=self.image_shape[1])])

        self.aug = augmentations
        self.data = list(self.data.items())
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

    def show(self, batch_idx: int) -> None:
        """
        This method showing image with label.

        :param batch_idx: batch number.
        """
        rows_columns_subplot = self.batch_size
        while np.math.sqrt(rows_columns_subplot) - int(np.math.sqrt(rows_columns_subplot)) != 0.0:
            rows_columns_subplot += 1
        rows_columns_subplot = int(np.math.sqrt(rows_columns_subplot))

        batch = self.data[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        plt.figure(figsize=(20, 20))
        for i, data_dict in enumerate(batch):
            class_name = data_dict[1]['class_name']
            image = cv2.cvtColor(cv2.imread(os.path.join(data_dict[0])), cv2.COLOR_BGR2RGB)
            image_copy = image
            image_augmented = self.aug(image=image_copy)
            plt.subplot(rows_columns_subplot, rows_columns_subplot, i+1)
            plt.imshow(image)
            plt.title('Original, class = "{}"'.format(class_name))
            plt.imshow(image_augmented['image'])
            plt.title('Augmented, class = "{}"'.format(class_name))
        if plt.waitforbuttonpress(0):
            plt.close('all')
            raise SystemExit
        plt.close()


def image_normalization(image: np.ndarray) -> np.ndarray:
    """
    Image normalization.
    :param image: image numpy array.
    :return: normalized image.
    """
    return image / 255.0


if __name__ == '__main__':
    x = DataGenerator(JSON_FILE_PATH, BATCH_SIZE, True, INPUT_SHAPE, NUMBER_OF_CLASSES)
