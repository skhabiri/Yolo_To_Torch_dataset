"""
takes the yolo format labeled images stored in ./images and ./labels directory
and crop each class and store it in standard torchvision dataset format.
Specifically, it creates a train and test directory and creates a folder for
each class and stores the crop class images randomly with a specified ratio in
test and train paths.

input yolo file structure:
./images: *.jpg
./labels: *.txt > [class_number x_center y_center bbox_width bbox_height]

output torchvision file structure:
./train: class_name1/*_cropped.jpg, class_name2/*_cropped.jpg, ...
./test: class_name1/*_cropped.jpg, class_name2/*_cropped.jpg, ...

* ex/
```
python ./yolo_To_torch_dataset.py -y ./yolo_format -t ./torch_dataset
                                -r 0.2 -wh 128 64
```
> -y, '--yolo': path to yolo format directory
> -t, '--torch': torchvision.dataset output directory
> -r, '--ratio': ratio of test to train split
> -wh, '--size': resize the image to a fixed width and height

"""

import re
import os
import glob
import argparse
import cv2
import matplotlib.pyplot as plt
import shutil
import numpy as np


def msg(name=None):
    return """
        python ./yolo_To_torch_dataset.py -y ./yolo_format -t ./torch_dataset
                                        -r 0.2 -wh 128 64
        """

parser = argparse.ArgumentParser(usage=msg())
parser.add_argument('-y', '--yolo', help='path to yolo format directory', dest="ypath")
parser.add_argument('-t', '--torch', nargs='?', help='torch dataset output directory', type=str,
                    const="./torch_dataset", default="./torch_dataset", dest="tpath")
parser.add_argument('-r', '--ratio', help='ratio of test to train split', dest="ratio", type=float)
parser.add_argument('-wh', '--size', nargs='+', help='desired cropped size, w, h', dest="size",
                    default=[64, 128], type=int)


# vars(obj) return obj.__dict__()
args = vars(parser.parse_args())

ypath = args["ypath"]
tpath = args["tpath"]
ratio = args["ratio"]
dim = tuple(args["size"])

# image and label path containing yolo labeled images
img_path = ypath + '/images'
label_path = ypath + '/labels'

# maps class number to human readable names
label_dict = {"0": "bird", "1": "flatwing", "2": "quadcopter"}

# create folders in destination torch folder for
# each class label in train and test
for k,v in label_dict.items():
    train_path = tpath + '/train/' + v
    test_path = tpath + '/test/' + v

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

def yolotxt(t):
    """
    convert a yolo formatted line of xywh to xyxy
    input: yolo .txt has [label, x_center, y_center, bbox_width, bbox_height]
    output: x_min, y_min, x_max, y_max
    """
    t0 = t.split()
    cls = t0[0]
    t1 = [float(i) for i in t0[1:]]
    xmin, ymin = t1[0] - 0.5 * t1[2], t1[1] - 0.5 * t1[3]
    xmax, ymax = xmin + t1[2], ymin + t1[3]
    return cls, xmin, ymin, xmax, ymax

count = 0

for filename in glob.glob(label_path + '/*.txt'):
    basename = os.path.basename(filename)
    basename_no_ext, extension = (os.path.splitext(basename)[i] for i in [0, 1])

    labelfile = open(filename,'r')
    lines = labelfile.readlines()

    for line in lines:

        img = cv2.imread(img_path + '/' + basename_no_ext + '.jpg')
        # open cv shape attribute is (height,width,channel)
        dh, dw, _ = img.shape

        cls, xmin, ymin, xmax, ymax = yolotxt(line)
        xmin = max(int(xmin * dw), 0)
        xmax = max(int(xmax * dw), 0)
        ymin = max(int(ymin * dh), 0)
        ymax = max(int(ymax * dh), 0)

        crop_img = img[ymin:ymax, xmin:xmax]

        # resize image to the requested size
        # pytorch CNN requires a minimum size dimensions
        resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

        if np.random.rand(1) < ratio:
            dir = '/test/'
        else:
            dir = '/train/'

        img_out = tpath + dir + label_dict[cls] + '/' + label_dict[cls] + basename_no_ext + '.jpg'

        cv2.imwrite(img_out, resized)
        count += 1
        if count % 500 == 0:
            print(f'{count} files created')
