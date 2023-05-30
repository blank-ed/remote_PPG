# This file houses algorithms that have not been segmented into their respective stages

import cv2
import numpy as np
from remote_PPG.filters import *
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


def extract_frames_yield(input_video):
    """
    param input_video:
        This method takes in a video file
    return:
        Yields the frames
    """

    cap = cv2.VideoCapture(input_video)
    success, image = cap.read()
    while success:
        yield image
        success, image = cap.read()
    cap.release()


def vj_face_detector(input_video, framework=None, width=1, height=1):
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

        if framework == 'PCA':
            red_values = np.sum(roi[:, :, 2], axis=(0, 1))
            green_values = np.sum(roi[:, :, 1], axis=(0, 1))
            blue_values = np.sum(roi[:, :, 0], axis=(0, 1))
            raw_sig.append([red_values, green_values, blue_values])
        elif framework == 'CHROM':
            filtered_roi = simple_skin_selection(roi)
            b, g, r, a = cv2.mean(filtered_roi)
            raw_sig.append([r, g, b])
        elif framework == 'ICA':
            b, g, r, a = cv2.mean(roi)
            raw_sig.append([r, g, b])
        elif framework == 'LiCVPR':
            b, g, r, a = cv2.mean(roi)
            raw_sig.append(g)

    return raw_sig


def raw_bg_signal(input_video, color='g'):
    """
    :param input_video:
        This takes in an input video file
    :param color:
        Select the background color channel to return.
        Default is green
    :return:
        Returns the mean RGB value of the background
    """

    raw_bg_sig = []

    model_path = 'Necessary_Files\\selfie_segmenter_landscape.tflite'
    base_options = BaseOptions(model_asset_path=model_path)

    mp_base_options = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create an image segmenter instance with the video mode:
    options = ImageSegmenterOptions(base_options=mp_base_options(model_asset_path=model_path),
                                    running_mode=VisionRunningMode.VIDEO, output_category_mask=True)

    frame_counter = 0
    fps = get_fps(input_video)

    with ImageSegmenter.create_from_options(options) as segmenter:
        for frame in extract_frames_yield(input_video):
            frame_time = int(frame_counter * (1000 / fps))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            segmented_masks = segmenter.segment_for_video(mp_image, frame_time)
            category_mask = segmented_masks.category_mask
            output = category_mask.numpy_view()

            output_mask_bool = np.where(output == 255, True, False)
            output_frame = np.zeros_like(frame)
            output_frame[output_mask_bool] = frame[output_mask_bool]

            output_mask_uint8 = output_mask_bool.astype(np.uint8)
            b, g, r, a = cv2.mean(frame, mask=output_mask_uint8)
            raw_bg_sig.append([r, g, b])

            frame_counter += 1

    if color == 'g':
        raw_bg_sig = [x[1] for x in raw_bg_sig]

    return raw_bg_sig


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
    for i in range(0, len(sig), int(increment * fps)):
        end = i + int(window_size * fps)
        if end >= len(sig):
            windowed_sig.append(sig[len(sig) - int(window_size * fps):len(sig)])
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

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return int(fps)

