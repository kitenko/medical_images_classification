from typing import Tuple

import tensorflow as tf
from classification_models.tfkeras import Classifiers

from config import NAME_MODEL, INPUT_SHAPE, NUMBER_OF_CLASSES, WEIGHTS


class CustomModelGit:
    def __init__(self, input_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES,
                 name_model: str = NAME_MODEL, weights: str = WEIGHTS) -> None:
        """
        You can get a specific model here.

        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param name_model: the name of the model to be built.
        :param weights: imageNet or None.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name_model = name_model
        self.weights = weights

    def build(self):
        """
        This function builds the model based on the model name.

        :return: tf.keras.model
        """
        name_model, preprocess_input = Classifiers.get(self.name_model)
        base_model = name_model(input_shape=self.input_shape, weights=self.weights, include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

        return model
