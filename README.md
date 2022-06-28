# Human Face Mask Object Recognition using CNNs

## Implementaion of Deep learning based Face Mask Detection with inference on an embedded NVIDIA Jetson device
### Academic Team Project

First we obtain the image from the monocular camera of NVIDIA Jetson Tx2 Board. 
This image is processed by a Yolo model to identify people in it and get their bounding boxes. 
These extracted bounding boxes act as the input for the second model which is responsible for identifying the faces and getting their bounding boxes, which are further passed to the third model -Resnet Model for classifying whether the face has a mask on or not.
After the detection, the distance of people from the camera is estimated by a distance estimation algorithm which uses bounding boxes obtained from the first model.

### Process Flow Diagram: </br>
<img align ="center" src ="https://user-images.githubusercontent.com/45971902/176203160-8c3b2f10-c10d-4d85-9ef3-e679966a138a.png">


### References: </br>
#### Model 1: </br>
YOLOv3 Standard - https://github.com/rohanrao619/Social_Distancing_with_AI/blob/master/Social_Distancing_Monitor.ipynb  </br>
YOLOv3 Tiny - https://github.com/siddharthbhonge/YOLO_with_Nvidia_jetson_TX2  </br>
#### Model 2: </br>
YOLO modified - https://github.com/sthanhng/yoloface </br>
HOG - https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/  </br>
#### Model 3: </br>
Resnet 50 - https://github.com/rohanrao619/Social_Distancing_with_AI/blob/master/Face_Mask_Classifier.ipynb </br>
#### Distance Estimation: </br>
https://github.com/CodinjaoftheWorld/Object-detection-using-yolov2-and-distance-estimation/blob/master/yolov2_Object_Detection_Distance_Estimation.ipynb  </br>
