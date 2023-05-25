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


def VJ_face_detector(input_video, framework=None, width=1, height=1):
    """
    :param input_video:
        This takes in an input video file
    :param framework:
        This is to specify the framework. Different frameworks have different ways of using the Viola-Jones
        face detector.
    :param ROI:
        Select the region of interest
            - BB: Bounding box of the whole face
            - FH: Forehead box using Viola Jones Face Detector
    :param width:
        Select the width of the detected face bounding box
    :param height:
        Select the height of the detected face bounding box
    :return:
        if framework == 'PCA':
            Returns the sum of RGB pixel values of video sequence from the ROI
        elif framework == 'CHROM':

        elif framework == 'ICA':
            Returns the averaged raw RGB signal from the ROI
    """

    raw_sig = []

    face_cascade = cv2.CascadeClassifier("Necessary_Files\\haarcascade_frontalface_default.xml")
    face_coordinates_prev = None
    frame_counter = 0
    for frame in extract_frames_yield(input_video):
        frame_counter += 1

        # Skip the first second as the camera have auto adjusting properties
        if frame_counter <= get_fps(input_video):
            continue

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

        if framework == 'PCA':
            red_values = np.sum(roi[:, :, 2], axis=(0, 1))
            green_values = np.sum(roi[:, :, 1], axis=(0, 1))
            blue_values = np.sum(roi[:, :, 0], axis=(0, 1))
            raw_sig.append([red_values, green_values, blue_values])
        elif framework == 'CHROM':
            # Apply simple skin selection thing for now just returning raw rgb values
            b, g, r, a = cv2.mean(roi)
            raw_sig.append([r, g, b])
        elif framework == 'ICA':
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

