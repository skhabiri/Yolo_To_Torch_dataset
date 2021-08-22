# Yolo_To_Torch_dataset
A utility to convert YOLO labeled images to torchvision dataset class

This utility takes the yolo format labeled images stored in ./images and ./labels directory
and crop each class and store it in torchvision dataset format. Specifically,
it creates a train and test directory with subdirectories for each class and
stores the crop images for each class randomly in test or train paths with a specified split ratio.
