import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
	help="path to facial landmark predictor",default ="/home/harishanth/GIT/CCTV/shape_predictor_68_face_landmarks.dat")

args = vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

K=0

while True:
    cap = cv2.VideoCapture('http://192.168.1.2/mjpg/video.mjpg')
    ret, frame = cap.read()


    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 2)
    print (len(rects))
    if len(rects) > 0:
        K = K + 1
    L = 0

    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        L = L + 1
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=96)
        faceAligned = fa.align(frame, gray, rect)

        import uuid

        f = str(uuid.uuid4())

        plt.imshow(cv2.cvtColor(faceOrig, cv2.COLOR_BGR2RGB))
        plt.title("original")
        plt.axis("off")
        plt.savefig('/home/harishanth/GIT/CCTV/Database/ %s face_Orig %s' % (K, L))
        #time.sleep(0.5)

        plt.imshow(cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB))
        plt.title("Aligned")
        plt.axis("off")
        plt.savefig('/home/harishanth/GIT/CCTV/Database/ %s face_Align %s' % (K, L))
        #time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()