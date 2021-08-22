# Yolo_To_Torch_dataset
A utility to convert YOLO labeled images to torchvision dataset class

This utility takes the yolo format labeled images stored in ./images and ./labels directory
and crop each class and store it in torchvision dataset format. Specifically,
it creates a train and test directory with subdirectories for each class and
stores the crop images for each class randomly in test or train paths with a specified split ratio.

* ex/
```
python ./yolo_To_torch_dataset.py -y ./yolo_format -t ./torch_dataset -r 0.2
```
> -y', '--yolo': path to yolo format directory
> -t', '--torch': torchvision.dataset output directory
> -r', '--ratio': ratio of test to train split
