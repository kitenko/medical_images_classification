import os
import json

from config import JSON_FILE_PATH_DATA_GEN, DATASET_PATH_IMAGES, JSON_FILE_PATH_INDEX_CLASS


def make_data_json(path_for_json_data_gen: str = JSON_FILE_PATH_DATA_GEN, data_image: str = DATASET_PATH_IMAGES,
                   proportion_test_images: float = 0.2, json_index_class: str = JSON_FILE_PATH_INDEX_CLASS) -> None:
    """
    This function creates json file with train and test data of images.

    :param path_for_json_data_gen: this is path where file will save.
    :param proportion_test_images: percentage of test images.
    :param data_image: path for data of images
    :param json_index_class: this is path where file will save.
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

    # create dictionary for data_generator
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

    # write json file for data_generator
    with open(path_for_json_data_gen, 'w') as f:
        json.dump(train_test_image_json, f, indent=4)

    # write json file with index class
    with open(json_index_class, 'w') as f:
        json.dump(numeric_index_class, f, indent=4)


if __name__ == '__main__':
    make_data_json(proportion_test_images=0.2)
