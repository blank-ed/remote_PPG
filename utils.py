# This file houses algorithms that have not been segmented into their respective stages

import cv2
import numpy as np

def extract_frames_yield(input_video):
    """
    param input_video:
        This method takes in a video file
    return:
        Yields the frames
    """
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()


def VJ_face_detector(input_video, width=0.6, height=1):
    """
    :param input_video:
        This takes in an input video file
    :param width:
        Select the width of the detected face bounding box
    :param height:
        Select the height of the detected face bounding box
    :return:
        Returns the raw RGB signal
    """

    raw_sig = []

    face_cascade = cv2.CascadeClassifier(r"C:\Users\ilyas\PycharmProjects\pythonProject1\remote_PPG\Necessary Files\haarcascade_frontalface_default.xml")
    face_coordinates_prev = None
    for frame in extract_frames_yield(input_video):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0 and face_coordinates_prev is not None:
            x, y, w, h = face_coordinates_prev
            x1 = int(x + (1 - width) / 2 * w)
            y1 = int(y + (1 - height) / 2 * h)
            x2 = int(x + (1 + width) / 2 * w)
            y2 = int(y + (1 + height) / 2 * h)
            roi = frame[y1:y2, x1:x2]

        else:
            for (x, y, w, h) in faces:
                face_coordinates_prev = (x, y, w, h)
                x1 = int(x + (1 - width) / 2 * w)
                y1 = int(y + (1 - height) / 2 * h)
                x2 = int(x + (1 + width) / 2 * w)
                y2 = int(y + (1 + height) / 2 * h)
                roi = frame[y1:y2, x1:x2]

        b, g, r, a = cv2.mean(roi)
        raw_sig.append([r, g, b])

    return raw_sig


def moving_window(sig, fps, window_size, increment):
    """
    :param sig:
        RGB signal
    :param fps:
        Frame rate of the video file (number of frames per second)
    :param window_size:
        Select the window size in seconds (s)
    :param increment:
        Select amount to be incremented in seconds (s)
    :return:
        returns the windowed signal
    """
    windowed_sig = []
    for i in range(0, len(sig), increment * fps):
        end = i + window_size * fps
        if end >= len(sig):
            windowed_sig.append(sig[len(sig) - window_size * fps:len(sig)])
            break
        windowed_sig.append(sig[i:end])
    return windowed_sig


def get_fps(input_video):
    """
    :param input_video:
        This takes in an input video file
    :return:
        Returns the fps of the video
    """
    vidcap = cv2.VideoCapture(input_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return int(fps)

