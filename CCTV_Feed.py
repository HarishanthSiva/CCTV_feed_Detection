import cv2

import urllib 
#from urllib.request import urlopen
import numpy as np


stream=urllib.urlopen('http://192.168.1.2/mjpg/video.mjpg')

bytes=''

while True:

    bytes+=stream.read(1024)

    a = bytes.find('\xff\xd8')

    b = bytes.find('\xff\xd9')

    if a!=-1 and b!=-1:

        jpg = bytes[a:b+2]

        bytes= bytes[b+2:]

        video = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)

        cv2.imshow('video',video)

        if cv2.waitKey(1) ==27:

            exit(0)   
