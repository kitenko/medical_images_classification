import argparse
import json

import cv2
import numpy as np
import tensorflow as tf

from classification_model_git import build_model
from config import INPUT_SHAPE, JSON_FILE_PATH_INDEX_CLASS


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    parser.add_argument('--path_video', type=str, default=None, help='Path for loading video for test.')
    return parser.parse_args()


def preparing_frame(image: np.ndarray, model) -> np.ndarray:
    """
    This function prepares the image and calls the predicted method.

    :param image: this is input image or frame.
    :param model: assembled model with loaded weights.
    :return: image with an overlay mask
    """
    image = cv2.resize(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    lable = model.predict(np.expand_dims(image, axis=0) / 255.0)[0]
    return image, lable


def get_key(d: dict, value: int) -> str:
    """
    This function returns the class name by index.

    :param d: dictionary with classes and indexes.
    :param value: predicted index by the model
    :return: key value by variable
    """
    for k, v in d.items():
        if v == value:
            return k


def visualization() -> None:
    """
    This function captures  video and resizes the image.
    """
    with open(JSON_FILE_PATH_INDEX_CLASS) as f:
        index = json.load(f)

    args = parse_args()
    model = build_model()
    model.load_weights('models_data/save_models/EfficientNetB0_imagenet_2021-05-18_00-39-10/EfficientNetB0.h5')

    cap = cv2.VideoCapture(args.path_video)
    cap = cv2.VideoCapture('test_video/vid_2.avi')

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.resize(frame, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
            image, index_class = preparing_frame(image=frame, model=model)
            class_image = get_key(index, np.argmax(index_class))
            image = cv2.resize(image, (720, 720))
            cv2.putText(image, class_image, (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(float("{0:.2f}".format(np.amax(index_class) * 100))) + '%', (500, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    visualization()
