# Mobile_Application_For_Visually_Impaired_People
Tracking Objects from live feed over a Voice Assistant
-------------------------------------------------------
* The main idea of this project is to enable users to track or locate an object and identifying theft
  This idea can be further customized into various areas like observing container loading sites and shopping malls where supervision is required
* we have used state of the art models that give the best results and have used only the objects relating to the coco dataset consisting of 80 classes only

Block 1
-------
Import all The required libraries and load all the pre-trained weights already trained on the coco dataset
also store all the class names in a list
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
net=cv2.dnn.readNetFromDarknet("yolo.cfg","yolov3-spp.weights")
classes=[]
with open('coco.names','r') as f:
    classes=[line.strip() for line in f.readlines()]
```
**Why Yolo ?**

they are a lot of models efficientdet, faster-RCNN, mobileNet which is the most scalable and light network but overall these, though Yolo is a multiplex network (i.e both very large in size and space) it is providing very accurate results comparatively. The biggest advantage of using YOLO is its superb speed – it's incredibly fast and can process 45 frames per second. YOLO also understands generalized object representation

[pratical comparision](https://www.youtube.com/watch?v=llBhBSgoWPs)

Block 2
--------
YOLO architecture study and extracting required outputs
--------------------------------------------------------
YOLO is an object detection model. using a single GPU using mini-batch size, YOLO achieves state-of-the-art results at a real-time speed on the MS COCO dataset with 43.5 % AP running at 65 FPS on a Tesla V100

Model architecture of YOLO
-----------------------------
![](https://miro.medium.com/max/825/1*jLUJU34dSbrRWdspJZbLXA.png)

Backbone
--------
Models such as ResNet, DenseNet, VGG, etc, are used as feature extractors. They are pre-trained on image classification datasets, like ImageNet, and then fine-tuned on the detection dataset

Neck
----
These are extra layers that go in between the backbone and head. They are used to extract different feature maps of different stages of the backbone. The neck part can be for example a FPN, PANet, Bi-FPN.

Head
----
* This is a network in charge of actually doing the detection part (classification and regression) of bounding boxes. A single output may look like (depending on the implementation): 4 values describing the predicted bounding box (x, y, h, w) and the probability of k classes + 1 (one extra for background). 
* Objected detectors anchor-based, like YOLO, apply the head network to each anchor box. Other popular one-stage detectors, which are anchor-based, are: Single Shot Detector[6] and RetinaNet[4]

NMS and Identification
----------------------
* The purpose of non-max suppression is to select the best bounding box for an object and reject or “suppress” all other bounding boxes. The NMS takes two things into account

1. The objectiveness score is given by the model
2. The overlap or IOU of the bounding boxes 
*** Intersection over Union(IOU) is an evaluation metric used to measure the accuracy of an object detector on a particular dataset
* It is the **intersection of overlap / intersection of union 
* In the numerator we compute the area of overlap between the predicted bounding box and the ground-truth bounding box.
* The denominator is the area of union, or more simply, the area encompassed by both the predicted bounding box and the ground-truth bounding box
* considering the final confidence threshold and NMS threshold the final bounding box with max confidence is selected and tagged with class name along with confidence score
