# People-Counter-Cpp
Count people in specific regions in images using Yolov5 (CPP, Libtorch)


## About the project
Count people inside some given ROIs. [Video](https://www.linkedin.com/posts/abdelrahman-othman-197235b7_cplusplus-opencv-deeplearning-activity-6902779631250874368-lrpw?utm_source=share&utm_medium=member_desktop)

![This is an image](https://github.com/Eng-Abdelrahman-M/People-Counter-Cpp/blob/master/frame_0099.jpg)


## Data
PETS 2009 Benchmark Data
Avaible here: http://cs.binghamton.edu/~mrldata/pets2009

## Model
- Yolov5 : https://github.com/Nebula4869/YOLOv5-LibTorch
- Pretrained model : https://github.com/deepakcrk/yolov5-crowdhuman
- Model was exported to torch script using https://github.com/ultralytics/yolov5/issues/251

## Configrations
We can change ROIs as we like in the config.yml file.
The output data can be found in "data/output" folder.

## Tools
- Visual studio 
