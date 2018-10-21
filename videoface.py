import cv2
import numpy as np


_FRAME_EXPANSION = 1.3
body_cascade = cv2.CascadeClassifier('haarcascade_upper_body.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

class VideoFace:
    def __init__(self, vs, window_width=None, window_height=None):
        self.vs = vs
        self.frame_width = 0
        self.frame_height = 0
        self.timeout = 0
        self.freeze = False
        self.window_width = window_width
        self.window_height = window_height
        self.ratio = window_height / window_width
        self.face_found = False
        self.adjust_factor = 5
        self.timeout_max = 100


    def frame_process(self):
        self.frame = self.vs.read()
        self.find_face()

    def find_face(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # minNeighbors 'specifies how many neightbors each candidate rectangle requires in order to keep it
        # Higher value less detections but with higher quality
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            self.timeout = 0
            w = 0
            h = 0
            x = 0
            y = 0
            self.face_found = True

            # use the largest face found
            for (_x, _y, _w, _h) in faces:
                if _w > w or _h > h:
                    w = _w
                    h = _h
                    x = _x
                    y = _y

            # If cropping is not frozen, adjust crop size
            if not self.freeze:
                if self.frame_width == 0:
                    # Frist time finding face set directly
                    self.center_x = x + w / 2
                    self.center_y = y + h / 2
                    self.frame_width = _FRAME_EXPANSION * w
                    self.frame_height = h
                    self.corner_x = x
                    self.cornder_y = y

                    # # Trigger on_found event
                    # if self.on_found is not None:
                    #     self.on_found()
                else:
                    # Adjust crop bounds increasing for greater difference
                    self.center_x += int((x + w / 2 - self.center_x) / self.adjust_factor)
                    self.center_y += int((y + h / 2 - self.center_y) / self.adjust_factor)
                    self.frame_width += int((1.5 * w - self.frame_width) / self.adjust_factor)
                    self.frame_height = int(self.frame_width * self.ratio)
                    self.corner_x = max(self.center_x - self.frame_width / 2, 0)
                    self.corner_y = max(self.center_y - self.frame_width / 2, 0)
        else:
            if self.timeout <= self.timeout_max:
                self.timeout += 1
            if self.timeout == self.timeout_max:
                self.timed_out()

        if self.face_found:
            # Crop frame
            if int(self.frame_width) > 30 and int(self.frame_height) > 30:
                self.cropped_frame = self.crop_image(self.frame, self.corner_x, self.corner_y,
                                                     self.frame_width, self.frame_height)
                self.cropped_frame = cv2.resize(self.cropped_frame, (self.window_width, self.window_height),
                                                interpolation=cv2.INTER_AREA)
                # # Trigger on_new_frame event
                # if self.on_new_frame is not None:
                #     self.on_new_frame(self.cropped_frame)

    def timed_out(self):
        self.face_found = False
        self.center_x = 0
        self.center_y = 0
        self.frame_width = 0
        self.frame_height = 0
        # if self.on_timeout is not None:
        #     self.on_timeout()

    @staticmethod
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
