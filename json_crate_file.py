import os
import json

from config import (JSON_FILE_PATH, DATASET_PATH_polyps, DATASET_PATH_retroflex_stomach,
                    DATASET_PATH_dyed_lifted_polyps, DATASET_PATH_pylorus,
                    DATASET_PATH_z_line, DATASET_PATH_bbps_0_1, DATASET_PATH_bbps_2_3,
                    DATA_PATH_cecum, DATASET_PATH_dyed_resection_margins)


def make_data_json(path_for_json: str = JSON_FILE_PATH, proportion_test_images: float = 0.2) -> None:
    """
    This function creates json file with train and test data of images.

    :param path_for_json: this is path where file will save.
    :param proportion_test_images: percentage of test images.
    """
    # directory list
    cecum = os.listdir(DATA_PATH_cecum)
    dyed_lifted_polyps = os.listdir(DATASET_PATH_dyed_lifted_polyps)
    dyed_resection_margins = os.listdir(DATASET_PATH_dyed_resection_margins)
    retroflex_stomach = os.listdir(DATASET_PATH_retroflex_stomach)
    bbps_2_3 = os.listdir(DATASET_PATH_bbps_2_3)
    polyps = os.listdir(DATASET_PATH_polyps)
    z_line = os.listdir(DATASET_PATH_z_line)
    bbps_0_1 = os.listdir(DATASET_PATH_bbps_0_1)
    pylorus = os.listdir(DATASET_PATH_pylorus)

    # create dictionary
    train_test_image_json = {'train': {}, 'test': {}}

    # create zip object
    path_name_label_zip = zip([DATA_PATH_cecum, DATASET_PATH_dyed_lifted_polyps, DATASET_PATH_dyed_resection_margins,
                               DATASET_PATH_retroflex_stomach, DATASET_PATH_bbps_2_3, DATASET_PATH_polyps,
                               DATASET_PATH_z_line, DATASET_PATH_bbps_0_1, DATASET_PATH_pylorus],
                              [cecum, dyed_lifted_polyps, dyed_resection_margins, retroflex_stomach, bbps_2_3,
                               polyps, z_line, bbps_0_1, pylorus],
                              ['cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'retroflex-stomach', 'bbps-2-3',
                               'polyps', 'z-line', 'bbps-0-1', 'pylorus'])

    # create full dict for json file
    for path_data, name_image, label in path_name_label_zip:
        for n, current_image_name in enumerate(name_image):
            if n < len(name_image) * proportion_test_images:
                train_test_image_json['test'][os.path.join(path_data, current_image_name)] = label
            else:
                train_test_image_json['train'][os.path.join(path_data, current_image_name)] = label

    # write json file
    with open(path_for_json, 'w') as f:
        json.dump(train_test_image_json, f, indent=4)


if __name__ == '__main__':
    make_data_json(proportion_test_images=0.2)