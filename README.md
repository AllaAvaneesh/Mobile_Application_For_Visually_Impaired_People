# Mobile_Application_For_Visually_Impaired_People
Tracking Objects from live feed over a Voice Assistant
-------------------------------------------------------
* The main idea of this project is to enable users to track or locate an object and identifying theft
  This idea can be further customized into various areas like observing container loading sites and shopping malls where supervision is required
* we have used state of the art models that give the best results and have used only the objects relating to the coco dataset consisting of 80 classes only

Block 1
-------
Import all the required libraries and load all the pre-trained weights already trained on the coco dataset
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

Block 3
--------
* Here after the architecture study we need to obtain the configuration file and also the weights file and understand what outputs are required for our functionality
* The standard yolo implementation by Joseph redmon returns coordinates of the center of the bounding boxes for the detected objects along with their width and height and also the confidence score along with their class 
but these values are in abstract form using some methods we can find the exact values 
1. The bounding box can be constructed from the coordinates obtained in the detections by using the height and width
2. The class_id from which class name is extracted can be obtained by finding the argmax of the score obtained from the detections
3. Using this class_id we can directly index for the confidence value
```
center_x=int(detection[0] * wt)
center_y=int(detection[1] * ht)
w=int(detection[2] * wt)
h=int(detection[3] * ht)
boxes.append([center_x,center_y,w,h])
confidences.append((float(confidence)))
classids.append(classid)
```
* After obtaining we can draw detections using opencv tools but there is a possibility that we can have more than one detection possible for the same class which better confidence
* In order to avoid this scenario we use Non-max suppression which suppresses the bounding box considering only the best one
*After Obtaining the Indices for unique detections we represent the boxes using opencv inbuilt functions such as rectangle, Circle, Text presenting the class name along with its confidence and also the center of the coordinates with x and y values mentioned

Block 4
-------
* During the process of block 3, we maintain two dictionaries object and object1 where object stores the center whereas object1 stores the top-left and bottom-right coordinates of the detected objects
* Our task here is to locate the object, for that we have defined some helper functions which calculates both point and line w.r.t point distance
```
def line_distance(a,b,c,x,y):
    return a*x+b*y+c/(np.sqrt(a**2+b**2))
```
```
def point_distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)
```    
* The line Distance function calculates the distance of the point from a line this is useful to check if the query object is close to the boundaries
* The Point Distance is the distance between two points which is used to track the nearest object

* The main turn here is that **the program always points to the nearest object or neglects the object if it is close to the boundaries**
if a query object is placed beside a small object then the program outputs the smaller object then we find it hard to find the small object itself 
* In order to face such challenges we have moved with a greedy approach considering the objects which are close to the query object but that of relatively large size
* Sorting by the ratio of its distance to the query object with its area will provide us relatively bigger object
```
distances[point_distance(v[0],v[1],point[0],point[1])/((coors[1][1]-coors[0][1])*(coors[1][0]-coors[0][0]))]=k
```
Object Instantiation and tracking
----------------------------------
* There will be many classes of the same type in the given scenario but to identify specific objects we require tracking They are complex methods out there but we focused to just store the labels in an efficient manner so to present each object with its unique object name
We have taken a default dictionary as it is not possible to handle if an entry is not present
The default dict is initialized with 0 so all the classes will have o objects detected before after that
we have updated the dictionary with the objects count and then later use this count to mention a unique number to 
each object in the scenario

Block 5
--------
* This block is devoted to directional sense
* After obtaining the object nearest to the query object we need to sense its direction four conditions can be chosen
1. If the x coordinate of the object lies outside the bounding box of the query object and must be greater than the query object and the y coordinate lies within the bounding box height it seems to be on the right
2.  If the x coordinate of the object lies outside the bounding box of the query object and must be less than the query object and the y coordinate lies within the bounding box height it seems to be on the left
3. Irrespective of the x coordinate if y coordinate lies above the bounding box it is said to be on the query object
4. finally if the y coordinate lies above the bounding box its is said to be under the query object
```
def direct(point1,point2,obj,height,width):
    if point1[0]>point2[0]+(width/2) and (point1[1]<=point2[1]+(height/2) and point1[1]>=point2[1]-(height/2)):
        print("It is to the right of {}".format(obj))
    elif point1[0]<point2[0]-(width/2) and (point1[1]<=point2[1]+(height/2) and point1[1]>=point2[1]-(height/2)):
        print("It is to the left of {}".format(obj))
    elif point1[1]<point2[1]+(height/2):
        print("It is on the {}".format(obj))
    elif point1[1]>point2[1]-(height/2):
        print("It is under the {}".format(obj))
    else:
        print("It is with the {}".format(obj))
```        
Block 6
--------
* Providing voice support to ease the process using voice commands
* Import the required packages and we are using pre-built audio present in windows10
* A voice assistant has the following capabilities
  1. speech recognition
  2. Intelligent Search
  3. customer engagement
  4. mood sensing
  5. simplifying processes
* we have designed a voice system named (KAREN) who comes with a lot of functionalities, he will greet you by recognizing time, he can open mail, browsers, he can play songs, he can even search for things

Block 7
--------
* After adding with these functionalities we combine all the modules together as a unit and include helper functions in the program that is connected to the voice system and finds any object when asked over speech
* The program responds out to the user with the nearest object along with its position with respect to the query object
* the service can be stopped by using the stop as a voice command 
* The program continuously monitors the objects and updates their locations if any objects are missed from the view the iteration will remove the object from the dictionary liberating not found message if queried

----------------
****The future scope of this project is to personalize the objects to the people present in the scenario and also to custom train this on a variety of objects****

**To run the modules on an independent machine and cover-up an API that provides service to the voice assistant from the LIVE feed tracking**

[Practical Implementation](https://drive.google.com/file/d/1y5Rw6geMx-1gqFu_oZ7n-LwgVJQNBqPC/view?usp=sharing)
