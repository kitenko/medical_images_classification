from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Activation, Add,
                                     MaxPool2D)


class CustomResNetModel:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        """

        :param input_shape:
        :param num_classes:
        """
        self._input_shape = input_shape
        self.num_classes = num_classes

    def build(self) -> tf.keras.Model:
        """

        :return:
        """
        inp = Input(shape=self._input_shape)
        conv2d = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', use_bias=False)(inp)
        batch_normalization = BatchNormalization()(conv2d)
        activation = Activation('relu')(batch_normalization)
        #
        maxpool2d = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(activation)
        #
        conv2d_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(maxpool2d)
        batch_normalization_1 = BatchNormalization()(conv2d_1)
        activation_1 = Activation('relu')(batch_normalization_1)
        conv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_1)
        batch_normalization_2 = BatchNormalization()(conv2d_2)
        add = Add()([maxpool2d, batch_normalization_2])
        #
        activation_2 = Activation('relu')(add)
        conv2d_2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)(activation_2)
        batch_normalization_3_1 = BatchNormalization()(conv2d_2_1)
        activation_3_1 = Activation('relu')(batch_normalization_3_1)
        conv2d_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_3_1)
        batch_normalization_3 = BatchNormalization()(conv2d_3)
        activation_3 = Activation('relu')(batch_normalization_3)
        conv2d_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_3)
        batch_normalization_4 = BatchNormalization()(conv2d_4)
        add_1 = Add()([activation_3_1, batch_normalization_4])
        #
        activation_4 = Activation('relu')(add_1)
        conv2d_5 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_4)
        batch_normalization_5 = BatchNormalization()(conv2d_5)
        activation_5 = Activation('relu')(batch_normalization_5)
        conv2d_6 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_5)
        batch_normalization_6 = BatchNormalization()(conv2d_6)
        add_2 = Add()([activation_4, batch_normalization_6])
        #
        activation_6 = Activation('relu')(add_2)
        conv2d_2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)(activation_6)
        batch_normalization_3_2 = BatchNormalization()(conv2d_2_2)
        activation_3_2 = Activation('relu')(batch_normalization_3_2)
        conv2d_7 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_3_2)
        batch_normalization_7 = BatchNormalization()(conv2d_7)
        activation_7 = Activation('relu')(batch_normalization_7)
        conv2d_8 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_7)
        batch_normalization_8 = BatchNormalization()(conv2d_8)
        add_3 = Add()([activation_3_2, batch_normalization_8])
        #
        activation_8 = Activation('relu')(add_3)
        conv2d_9 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_8)
        batch_normalization_9 = BatchNormalization()(conv2d_9)
        activation_9 = Activation('relu')(batch_normalization_9)
        conv2d_10 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_9)
        batch_normalization_10 = BatchNormalization()(conv2d_10)
        add_4 = Add()([activation_8, batch_normalization_10])
        #
        activation_10 = Activation('relu')(add_4)
        conv2d_8_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)(activation_10)
        batch_normalization_9_1 = BatchNormalization()(conv2d_8_1)
        activation_10_1 = Activation('relu')(batch_normalization_9_1)
        conv2d_11 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_10_1)
        batch_normalization_11 = BatchNormalization()(conv2d_11)
        activation_11 = Activation('relu')(batch_normalization_11)
        conv2d_12 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_11)
        batch_normalization_12 = BatchNormalization()(conv2d_12)
        add_5 = Add()([activation_10_1, batch_normalization_12])
        #
        activation_12 = Activation('relu')(add_5)
        conv2d_13 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_12)
        batch_normalization_13 = BatchNormalization()(conv2d_13)
        activation_13 = Activation('relu')(batch_normalization_13)
        conv2d_14 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_13)
        batch_normalization_14 = BatchNormalization()(conv2d_14)
        add_6 = Add()([activation_12, batch_normalization_14])
        #
        activation_14 = Activation('relu')(add_6)
        conv2d_8_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)(activation_14)
        batch_normalization_14_1 = BatchNormalization()(conv2d_8_2)
        activation_14_1 = Activation('relu')(batch_normalization_14_1)
        conv2d_15 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_14_1)
        batch_normalization_15 = BatchNormalization()(conv2d_15)
        activation_15 = Activation('relu')(batch_normalization_15)
        conv2d_16 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_15)
        batch_normalization_16 = BatchNormalization()(conv2d_16)
        add_7 = Add()([activation_14_1, batch_normalization_16])
        #
        activation_16 = Activation('relu')(add_7)
        conv2d_17 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_16)
        batch_normalization_17 = BatchNormalization()(conv2d_17)
        activation_17 = Activation('relu')(batch_normalization_17)
        conv2d_18 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(activation_17)
        batch_normalization_18 = BatchNormalization()(conv2d_18)
        add_8 = Add()([activation_16, batch_normalization_18])
        activation_18 = Activation('relu')(add_8)
        #
        x = GlobalAveragePooling2D()(activation_18)
        x = Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inp, outputs=x)
        return model
