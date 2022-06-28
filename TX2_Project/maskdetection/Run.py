#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("..")
import os
import math
import numpy as np
import cv2
from ctypes import *
import face_detection 
# from yoloface.yolo.yolo import YOLO,detect_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from PIL import ImageDraw, Image
from tensorflow.python.framework.ops import disable_eager_execution
from HumanDetection import detect
from FaceDetHOG import face_detection
disable_eager_execution()

import time
start_time = time.time()
# Function to calculate the distance of object from the camera lense
def dist_calculator(startX,startY,endX,endY,box_width,box_height,img_w,img_h):
    x_3,y_3 = startX, endY - (box_height/7) # top left of the triangle
    #assumption: camera is rasied above the ground so considering 90% of the height of the image height
    x_1,y_1 = img_w/2,0.9*img_h # bottom of the triangle
    x_2,y_2 = endX , endY - (box_height/7) # top right of the triangle

    #find the angle between bottom and right point
    angle_x1_x2 = math.degrees(math.atan2(x_1-x_2,y_1-y_2))
    #find the angle between bottom and left point
    angle_x1_x3 = math.degrees(math.atan2(x_1-x_3,y_1-y_3))

    angle_right = 90 + angle_x1_x2
    angle_left = 90 - angle_x1_x3

    #total angle of view for the car from bottom center point of the image.
    total_angle = angle_right + angle_left

    # Bench length assumed to be 2 meters in millimeters. This value can automated, based on the type of bench used.
    bench_length = 600.0    # Average Human Stride Width
    # bench_length = box_width * 1.4   # Average Human Stride Width
    # print("Box Width :")
    # print(box_width)
    # print("Box Length :")
    # print(bench_length)
    

    #distance to object = (size of object) x (1°/angular size in degrees) x 57
    #Refer the link for more understadnign on the formula mentioned above - https://www.cfa.harvard.edu/webscope/activities/pdfs/measureSize.PDF
    distance = (bench_length*(1.0/total_angle)*57) / 100

    return total_angle,distance

class yolo_args:
    pass

def get_args():
    
    args = yolo_args()
    args.model = "../yoloface/model-weights/YOLO_Face.h5"
    args.anchors = "../yoloface/cfg/yolo_anchors.txt"
    args.classes = "../yoloface/cfg/face_classes.txt"
    args.score = 0.5
    args.iou = 0.45
    args.img_size = (416,416)
    args.image = "../yoloface/samples/outside_000001.jpg"
    return args

args = get_args()

# yolo = YOLO(args)

BASE_PATH = "./"
FILE_PATH = "test_video.mp4"
IMAGE_PATH = BASE_PATH + "TestImage.jpg"

# Initialize a Face Detector 
# Confidence Threshold can be Adjusted, Greater values would Detect only Clear Faces
# detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# Load Pretrained Face Mask Classfier (Keras Model)
mask_classifier = load_model("./Models/ResNet50_Classifier.h5")


# Load YOLOv3
# net = cv2.dnn.readNet(BASE_PATH+"Models/"+"yolov3.weights", BASE_PATH+"Models/"+"yolov3.cfg")

# Load COCO Classes
classes = []
with open(BASE_PATH+"Models/"+"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
##################################### Analyze ################################################



def analyse_image(img):
    image_recieved_time = time.time()
    print("Image Recieved............................................................................")
    # Get Frame Dimentions
    height, width, channels = img.shape

    # Detect Objects in the Frame with YOLOv3
    # blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # outs = net.forward(output_layers)
    outs = detect(img)
    model_1_time = time.time()
    print("Person Detection Output --- %s seconds ---" % (model_1_time - image_recieved_time))
    class_ids = []
    confidences = []
    boxes = []

    # Store Detected Objects with Labels, Bounding_Boxes and their Confidences
    # for out in outs:
    for detection in outs:
        # scores = detection[5:]
        # scores = detection[1]
        # class_id = np.argmax(scores)
        class_id = detection[0] 
        confidence = detection[1]
        pstring = class_id+": "+str(np.rint(100 * confidence))+"%"  
        if confidence > 0.5:
            # print(pstring)
            bounds = detection[2]  
            h = int(bounds[3])  
            w = int(bounds[2])
            # Get Center, Height and Width of the Box
            # center_x = int(detection[0] * width)
            # center_y = int(detection[1] * height)
            # w = int(detection[2] * width)
            # h = int(detection[3] * height)

            # Topleft Co-ordinates
            # x = int(center_x - w / 2)
            # y = int(center_y - h / 2)
            x = int(bounds[0] - bounds[2]/2)  
            y = int(bounds[1] - bounds[3]/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize empty lists for storing Bounding Boxes of People and their Faces
    persons = []
    masked_faces = []
    unmasked_faces = []

    # Work on Detected Persons in the Frame
    for i in range(len(boxes)):
        if i in indexes:

            box = np.array(boxes[i])
            box = np.where(box<0,0,box)
            (x, y, w, h) = box

            label = class_ids[i]
            
            model_2_latency = 0
            model_3_latency = 0

            if label=='person':
                # print("Person Found")
                persons.append([x,y,w,h])
                
                # Save Image of Cropped Person (If not required, comment the command below)
                t_image_path = BASE_PATH + "Results/Extracted_Persons/Person"+"_"+str(len(persons))+".jpg"
                #cv2.imwrite(t_image_path,img[y:y+h,x:x+w])

                # Detect Face in the Person
                person_rgb = img[y:y+h,x:x+w,::-1]   # Crop & BGR to RGB
                # print("Face Detector Operation Start")
                face_det_time_start = time.time()
                detections = face_detection(img[y:y+h,x:x+w])
                # res_image, detections = yolo.detect_image(Image.fromarray(img[y:y+h,x:x+w,::-1]))           
                # print("Face Detector Operation End")
                model_2_time = time.time()
                model_2_latency = model_2_latency + (model_2_time - face_det_time_start)
                model_3_time = 0
                if i == len(boxes)-1:
                    # model_2_time = time.time()
                    print("Face Detection Output --- %s seconds ---" % model_2_latency)
                # If a Face is Detected
                if detections.shape[0] > 0:
                    # print("Face FOUND")
                    detection = np.array(detections[0])
                    detection = np.where(detection<0,0,detection)

                    # Calculating Co-ordinates of the Detected Face
                    x1 = x + int(detection[0])
                    x2 = x + int(detection[2])
                    y1 = y + int(detection[1])
                    y2 = y + int(detection[3])

                    
                    face_rgb = img[y1:y2,x1:x2,::-1]   

                    # Preprocess the Image
                    face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    face_arr = preprocess_input(face_arr)
                    
                    mask_det_time_start = time.time()
                    # Predict if the Face is Masked or Not
                    score = mask_classifier.predict(face_arr)
                    model_3_time = time.time()
                    model_3_latency = model_3_latency + (model_3_time - mask_det_time_start)
                    
                    # Determine and store Results
                    if score[0][0]<0.5:
                        masked_faces.append([x1,y1,x2,y2])
                    else:
                        unmasked_faces.append([x1,y1,x2,y2])

                    cropped_face_path = BASE_PATH + "Results/Extracted_Faces/Person"+"_"+str(len(persons))+".jpg"
                    # Save Image of Cropped Face (If not required, comment the command below)
                    # cv2.imwrite(cropped_face_path,img[y1:y2,x1:x2])
                if i == len(boxes)-1:
                    print("Mask Classification Output --- %s seconds ---" % (model_3_latency))
    # Calculate Coordinates of People Detected and find Clusters using DBSCAN
    person_coordinates = []

        

    # Count 
    person_count = len(persons)
    masked_face_count = len(masked_faces)
    unmasked_face_count = len(unmasked_faces)


    # Put Bounding Boxes on People in the Frame
    for p in range(person_count):
        a,b,c,d = persons[p]
        cv2.rectangle(img, (a, b), (a + c, b + d), (0,255,0), 2)
        # Green if Safe, Red if UnSafe
        _,distance = dist_calculator(a,b,a + c,b + d,c,d,width,height)
        text = "Distance: {} m".format(round(distance,4))
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = a
        text_offset_y = b + d
        # set the rectangle background to white
        rectangle_bgr = (0, 0, 0)
        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)



    # Put Bounding Boxes on Faces in the Frame
    # Green if Safe, Red if UnSafe
    for f in range(masked_face_count):

        a,b,c,d = masked_faces[f]
        cv2.rectangle(img, (a, b), (c,d), (0,255,0), 2)

    for f in range(unmasked_face_count):

        a,b,c,d = unmasked_faces[f]
        cv2.rectangle(img, (a, b), (c,d), (0,0,255), 2)

    # Show Monitoring Status in a Black Box at the Top
    cv2.rectangle(img,(0,0),(width,50),(0,0,0),-1)
    cv2.rectangle(img,(1,1),(width-1,50),(255,255,255),2)

    xpos = 15

    string = "Total People = "+str(person_count)
    cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]


    string = "( " +str(masked_face_count)+" Masked "+str(unmasked_face_count)+" Unmasked "+ str(person_count-masked_face_count-unmasked_face_count)+" Unknown )"
    cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    image_done_time = time.time()
    print("Final Output --- %s seconds ---" % (image_done_time - image_recieved_time))
    print("..........................................................................................")
    return img



## Tegra cam
WINDOW_NAME = 'TX2 Mask Detector'
import subprocess
def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=0 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

def read_cam(cap):
    show_help = True
    full_scrn = False

    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read() # grab the next image frame from camera
        # img = cv2.imread(IMAGE_PATH) 
        img = analyse_image(img)
        # Save the Frame in frame_no.png format (If not required, comment the command below)
        # cv2.imwrite(BASE_PATH+"Results/Frames/Person.jpg",img)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

def main():

    print('OpenCV version: {}'.format(cv2.__version__))
    image_width = 1920
    image_height = 1080
    cap = open_cam_onboard(image_width,image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(image_width, image_height)
    read_cam(cap)

    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()

if __name__ == '__main__':
    main()
