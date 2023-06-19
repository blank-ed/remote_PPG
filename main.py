import cv2
import os

from remote_PPG.filters import simple_skin_selection
from remote_PPG.utils import extract_frames_yield

width = 1
height = 1


file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'remote_PPG', 'Necessary_Files', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(file_path)

face_coordinates_prev = None
mp_coordinates_prev = None
frame_count = 0

for frame in extract_frames_yield(r'C:\Users\Admin\Desktop\LGI-PPG Dataset\LGI_PPGI\talk\3cpi_talk\cv_camera_sensor_stream_handler.avi'):
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) != 0:
        print(frame_count)

    # Look through the first 30 frames until face is detected
    # if len(faces) == 0 and frame_count <= 30:
    #     continue

    # if (len(faces) == 0 or len(faces) > 1) and face_coordinates_prev is not None:
    #     x, y, w, h = face_coordinates_prev
    #     x1 = int(x + (1 - width) / 2 * w)
    #     y1 = int(y + (1 - height) / 2 * h)
    #     x2 = int(x + (1 + width) / 2 * w)
    #     y2 = int(y + (1 + height) / 2 * h)
    #     roi = frame[y1:y2, x1:x2]
    #
    # else:
    #     for (x, y, w, h) in faces:
    #         face_coordinates_prev = (x, y, w, h)
    #         x1 = int(x + (1 - width) / 2 * w)
    #         y1 = int(y + (1 - height) / 2 * h)
    #         x2 = int(x + (1 + width) / 2 * w)
    #         y2 = int(y + (1 + height) / 2 * h)
    #         roi = frame[y1:y2, x1:x2]
    #
    # filtered_roi = simple_skin_selection(roi)
    # b, g, r, a = cv2.mean(filtered_roi)
