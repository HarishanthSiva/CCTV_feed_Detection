
#python Real_time_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat



import cv2

import urllib 
#from urllib.request import urlopen
import numpy as np

import matplotlib.pyplot as plt
#from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
	help="path to facial landmark predictor",default ="/home/harishanth/GIT/CCTV/shape_predictor_68_face_landmarks.dat")

args = vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

stream = urllib.urlopen('http://192.168.1.2/mjpg/video.mjpg')
#stream= cv2.VideoCapture('http://192.168.1.2/mjpg/video.mjpg')


bytes=''

while True:

    #time.sleep(1)
    #stream = urllib.urlopen('http://192.168.1.2/mjpg/video.mjpg')

    bytes += stream.read(1024)


    a = bytes.find('\xff\xd8')

    b = bytes.find('\xff\xd9')

    if a!=-1 and b!=-1:

        jpg = bytes[a:b+2]

        bytes= bytes[b+2:]

        frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)

        cv2.imshow('video',frame)
	    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('video', frame)

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(frame, 2)

        print (len(rects))

	    #time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #time.sleep(1)
cv2.destroyAllWindows()
vs.stop()