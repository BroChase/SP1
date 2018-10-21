from videoface import VideoFace
from imutils.video import VideoStream
import time
import cv2
import numpy as np


body_cascade = cv2.CascadeClassifier('haarcascade_upper_body.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
_WINDOW_WIDTH = 300
_WINDOW_HEIGHT = 300
_FRAME_EXPANSION = .7


def timeout():
    print("Face was lost for timeout period")

def crop_image(image, x, y, w, h):
    iw = np.size(image, 1)
    ih = np.size(image, 0)
    # Ensure crop fits in frame
    w = int(max(min(iw, w), 40))
    h = int(max(min(ih, h), 50))
    # Prevent out of bounds
    x = int(max(min(iw - w, x), 0))
    y = int(max(min(ih - h, y), 0))
    return image[y:y + h, x:x + w]


def timed_out():
    face_found = False
    center_x = 0
    center_y = 0
    frame_width = 0
    frame_height = 0

if __name__ == '__main__':
    print("Warming Camera...")
    # vs = VideoStream().start()
    cap = cv2.VideoCapture(0)

    frame_width = 0
    frame_height = 0
    timeout = 0
    freeze = False
    window_width = _WINDOW_WIDTH
    window_height = _WINDOW_HEIGHT
    ratio = window_height / window_width
    face_found = False
    adjust_factor = 5
    timeout_max = 100
    center_x = 0
    center_y = 0
    corner_x = 0
    corner_y = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # minNeighbors 'specifies how many neightbors each candidate rectangle requires in order to keep it
        # Higher value less detections but with higher quality
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            timeout = 0
            w = 0
            h = 0
            x = 0
            y = 0
            face_found = True

            # use the largest face found
            for (_x, _y, _w, _h) in faces:
                if _w > w or _h > h:
                    w = _w
                    h = _h
                    x = _x
                    y = _y

            # If cropping is not frozen, adjust crop size
            if not freeze:
                if frame_width == 0:
                    # Frist time finding face set directly
                    center_x = x + w / 2
                    center_y = y + h / 2
                    frame_width = _FRAME_EXPANSION * w
                    frame_height = h
                    corner_x = x
                    cornder_y = y
                else:
                    # Adjust crop bounds increasing for greater difference
                    center_x += int((x + w / 2 - center_x) / adjust_factor)
                    center_y += int((y + h / 2 - center_y) / adjust_factor)
                    frame_width += int((1.5 * w - frame_width) / adjust_factor)
                    frame_height = int(frame_width * ratio)
                    corner_x = max(center_x - frame_width / 2, 0)
                    corner_y = max(center_y - frame_width / 2, 0)
        else:
            if timeout <= timeout_max:
                timeout += 1
            if timeout == timeout_max:
                timed_out()

        if face_found:
            # Crop frame
            if int(frame_width) > 30 and int(frame_height) > 30:
                cropped_frame = crop_image(frame, corner_x, corner_y,
                                                     frame_width, frame_height)
                cropped_frame = cv2.resize(cropped_frame, (window_width, window_height),
                                                interpolation=cv2.INTER_AREA)

                cv2.imshow('img', cropped_frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
