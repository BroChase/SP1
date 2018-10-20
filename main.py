import cv2

import numpy as np
import imutils
from PIL import Image
from PIL import ImageTk

body_cascade = cv2.CascadeClassifier('haarcascade_upper_body.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
cap = cv2.VideoCapture(0)
var = 1
# point = []
while True:
    point = []
    # Ret is a bool if an image has been retrieved or not.
    ret, img = cap.read()

    # converts the img from BlueGreenRed 'BGR' to Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gray scaled image with a 'Step factor=1.4' for image rescaling to attempt to match a face
    # minNeighbors 'specifies how many neightbors each candidate rectangle requires in order to keep it
    # Higher value less detections but with higher quality
    body = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(200, 200),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

    for(x, y, w, h) in faces:
        # Get center point of image (p, n) coord point
        n = (y + h)/2
        p = (x + w)/2
        # Save the point
        point.append((n, p))
        x = int(point[0][0] + 100)
        y = int(point[0][1] - 130)
        # On the img colored rectangle is drawn
        cv2.rectangle(img, (x, y), (x+200, y+260), (255, 0, 0), 3)
        # Crop image to face
        roi = img[y:y+260, x:x+200]
        # save crop image todo to DS queue/orstack
        # cv2.imwrite("roi.jpg", roi)

    # for(x, y, w, h) in body:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # for (x, y, w, h) in profile:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # frame = frame[ybig:ybig + hbig, xbig:xbig + wbig]
    # frame = imutils.resize(frame, width=300)
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    # image = ImageTk.PhotoImage(image)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
