# Yolo_To_Torch_dataset
A utility to convert YOLO labeled images to torchvision dataset class

This utility takes the yolo format labeled images stored in ./images and ./labels directory
and crop each class and store it in torchvision dataset format. Specifically,
it creates a train and test directory and creates a folder for each class and
stores the crop class images randomly with a specified ratio in test and train
paths.

* ex/
```
python ./yolo_To_torch_dataset.py -y ./yolo_format -t ./torch_dataset -r 0.2 -wh 128 64
```
 > -y, '--yolo': path to yolo format directory
 > 
 > -t, '--torch': torchvision.dataset output directory
 >
 > -r, '--ratio': ratio of test to train split
 >
 > -wh, '--size': resize the image to a fixed width and height
