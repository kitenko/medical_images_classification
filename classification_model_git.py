from typing import Tuple

import tensorflow as tf
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers

from config import NAME_MODEL, INPUT_SHAPE, NUMBER_OF_CLASSES, WEIGHTS


def build_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES,
                name_model: str = NAME_MODEL, weights: str = WEIGHTS) -> tf.keras.models.Model:
    """
    This function creates a model from 'classification_models.tf.keras' or 'efficientnet.tf.keras' library depending on
    the input parameters.

    :param input_shape: input shape (height, width, channels).
    :param num_classes: number of classes.
    :param name_model: the name of the model to be built.
    :param weights: ImageNet or None.
    :return: tf.keras.models.Model
    """

    if NAME_MODEL[:-2].lower() != 'efficientnet':
        name_model, preprocess_input = Classifiers.get(name_model)
        base_model = name_model(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB0':
        base_model = efn.EfficientNetB0(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB1':
        base_model = efn.EfficientNetB1(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB2':
        base_model = efn.EfficientNetB2(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB3':
        base_model = efn.EfficientNetB3(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB4':
        base_model = efn.EfficientNetB4(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB5':
        base_model = efn.EfficientNetB5(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB6':
        base_model = efn.EfficientNetB6(input_shape=input_shape, weights=weights, include_top=False)
    elif name_model == 'EfficientNetB7':
        base_model = efn.EfficientNetB7(input_shape=input_shape, weights=weights, include_top=False)
    else:
        base_model = efn.EfficientNetL2(input_shape=input_shape, weights=weights, include_top=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model
