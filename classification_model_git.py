from typing import Tuple

import tensorflow as tf
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers

from config import NAME_MODEL, INPUT_SHAPE, NUMBER_OF_CLASSES, WEIGHTS


class CustomModelGit:
    def __init__(self, input_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES,
                 name_model: str = NAME_MODEL, weights: str = WEIGHTS) -> None:
        """
        This class creates a model from 'classification_models.tfkeras' or 'efficientnet.tfkeras' library depending on
        the input parameters.

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

    def build_efficientnet(self) -> tf.keras.model:
        """
        This function builds efficientnet model based on the model name.

        :return: tf.keras.model
        """
        if self.name_model == 'EfficientNetB0':
            base_model = efn.EfficientNetB0(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB1':
            base_model = efn.EfficientNetB1(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB2':
            base_model = efn.EfficientNetB2(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB3':
            base_model = efn.EfficientNetB3(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB4':
            base_model = efn.EfficientNetB4(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB5':
            base_model = efn.EfficientNetB5(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB6':
            base_model = efn.EfficientNetB6(input_shape=self.input_shape, weights=self.weights, include_top=False)
        elif self.name_model == 'EfficientNetB7':
            base_model = efn.EfficientNetB7(input_shape=self.input_shape, weights=self.weights, include_top=False)
        else:
            base_model = efn.EfficientNetL2(input_shape=self.input_shape, weights=self.weights, include_top=False)

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

        return model