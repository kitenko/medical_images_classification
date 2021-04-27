import os
os.getcwd()
collection = "/home/andre/pycharm/classification_endoscopic_image/data/z-line/"
for i, filename in enumerate(os.listdir(collection)):
    os.rename(collection + filename, collection + "z-line_" + str(i) + ".jpg")