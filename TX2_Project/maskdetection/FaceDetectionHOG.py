from Social_Distancing_with_AI.HumanDetection import detect
from imutils import face_utils
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()

def face_detection(image):

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.

    # Get faces from image
    rects = detector(gray, 0)
    detections = np.empty((0, 4))
    # For each detected face, draw boxes.
    for (i, rect) in enumerate(rects):
        # Finding points for rectangle to draw on face
        x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
            1, rect.bottom() + 1, rect.width(), rect.height()
        detections = np.append(detections,[[x1, y1, x2, y2]],axis=0)
    
    return detections