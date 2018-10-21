import time
import cv2
from imgcapture import VideoStream


body_cascade = cv2.CascadeClassifier('haarcascade_upper_body.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
_WINDOW_WIDTH = 300
_WINDOW_HEIGHT = 300
_FRAME_EXPANSION = 1.3


if __name__ == '__main__':
    print("Warming Camera...")
    time.sleep(2)
    VideoStream(_WINDOW_WIDTH, _WINDOW_HEIGHT, _FRAME_EXPANSION)
