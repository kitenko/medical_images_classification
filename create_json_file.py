import os
import json

from config import JSON_FILE_PATH, DATASET_PATH_IMAGES


def make_data_json(path_for_json: str = JSON_FILE_PATH, data_image: str = DATASET_PATH_IMAGES,
                   proportion_test_images: float = 0.2) -> None:
    """
    This function creates json file with train and test data of images.

    :param path_for_json: this is path where file will save.
    :param proportion_test_images: percentage of test images.
    :param data_image: path for data of images
    """

    path_images, name_images, label_images = [], [], []
    numeric_index_class = {}

    for j, i in enumerate(os.listdir(data_image)):
        try:
            path_images.append(os.path.join(data_image, i))
            name_images.append(os.listdir(os.path.join(data_image, i)))
            label_images.append(i)
            numeric_index_class[i] = j
        except NotADirectoryError:
            print('File "data.json existed, but it was reconfigured."')

    path_name_label_zip = zip(path_images, name_images, label_images)

    # create dictionary
    train_test_image_json = {'train': {}, 'test': {}}

    # create full dict for json file
    for path_data, name_image, label in path_name_label_zip:
        for n, current_image_name in enumerate(name_image):
            if n < len(name_image) * proportion_test_images:
                train_test_image_json['test'][os.path.join(path_data, current_image_name)] = {
                 'class_name': label,
                 'index': numeric_index_class[label]
                }
            else:
                train_test_image_json['train'][os.path.join(path_data, current_image_name)] = {
                 'class_name': label,
                 'index': numeric_index_class[label]
                }
    # write json file
    with open(path_for_json, 'w') as f:
        json.dump(train_test_image_json, f, indent=4)


if __name__ == '__main__':
    make_data_json(proportion_test_images=0.2)
