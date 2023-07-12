# This file houses algorithms that have not been segmented into their respective stages

from remote_PPG.filters import *
import mediapipe as mp
import os
from mediapipe.tasks.python import vision, BaseOptions
from remote_PPG.utils import *


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


def extract_raw_sig(input_video, framework=None, ROI_type='None', width=1, height=1, pixel_filtering=True):
    """
    :param input_video:
        This takes in an input video file
    :param framework:
        This is to specify the framework. Different frameworks have different ways of extracting raw RGB signal
    :param ROI:
        Select the region of interest
            - BB: Bounding box of the whole face
            - FH: Forehead box using Viola Jones Face Detector
    :param width:
        Select the width of the detected face bounding box
    :param height:
        Select the height of the detected face bounding box
    :param pixel_filtering:
        Apply a simple skin selection process through pixel filtering
    :return:
        if framework == 'PCA':
            Returns the sum of RGB pixel values of video sequence from the ROI
        elif framework == 'CHROM':

        elif framework == 'ICA':
            Returns the averaged raw RGB signal from the ROI
    """

    if 'subject31' in input_video:
        scaleFactor = 1.3
    else:
        scaleFactor = 1.1

    raw_sig = []

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                   min_detection_confidence=0.5)

    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'remote_PPG', 'Necessary_Files', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(file_path)

    face_coordinates_prev = None
    mp_coordinates_prev = None
    mask = None
    frame_count = 0
    usable_roi_count = 0

    # for frame in extract_frames_yield(input_video):
    for frame in input_video:
        frame = cv2.imread(frame)
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # Look through the first 200 frames until face is detected
        if len(faces) == 0 and frame_count <= 200:
            continue

        if (len(faces) == 0 or len(faces) > 1) and face_coordinates_prev is not None:
            x, y, w, h = face_coordinates_prev
            x1 = int(x + (1 - width) / 2 * w)
            y1 = int(y + (1 - height) / 2 * h)
            x2 = int(x + (1 + width) / 2 * w)
            y2 = int(y + (1 + height) / 2 * h)
            roi = frame[y1:y2, x1:x2]

        else:
            for (x, y, w, h) in faces:
                usable_roi_count += 1
                face_coordinates_prev = (x, y, w, h)
                x1 = int(x + (1 - width) / 2 * w)
                y1 = int(y + (1 - height) / 2 * h)
                x2 = int(x + (1 + width) / 2 * w)
                y2 = int(y + (1 + height) / 2 * h)
                roi = frame[y1:y2, x1:x2]

        if framework == 'LiCVPR':
            results = mp_face_mesh.process(roi)
            region = roi
        else:
            results = mp_face_mesh.process(frame)
            region = frame

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                if framework == 'GREEN':
                    selected_landmarks = [67, 299, 296, 297, 10]
                elif framework == 'LiCVPR':
                    selected_landmarks = [234, 132, 136, 152, 365, 361, 454, 380, 144]
                else:
                    selected_landmarks = [0]

                selected_coordinates = [
                    (int(landmarks[i].x * region.shape[1]), int(landmarks[i].y * region.shape[0])) for i in
                    selected_landmarks]
                mp_coordinates_prev = selected_coordinates

        else:
            if mp_coordinates_prev is not None:  # Check if mp_coordinates_prev is not None
                selected_coordinates = mp_coordinates_prev
            else:
                continue

        if framework == 'PCA':
            red_values = np.sum(roi[:, :, 2], axis=(0, 1))
            green_values = np.sum(roi[:, :, 1], axis=(0, 1))
            blue_values = np.sum(roi[:, :, 0], axis=(0, 1))
            raw_sig.append([red_values, green_values, blue_values])

        elif framework == 'CHROM' or framework == 'POS' or framework == 'ICA':
            region_of_interest = roi

        elif framework == 'PhysNet':
            if usable_roi_count == 129:
                break
            resized_roi = cv2.resize(roi, (128, 128))
            raw_sig.append(resized_roi)

        elif framework == 'DeepPhys':
            # U have two types of ROI. Implement them. For now just use ROI from VJ
            downsampled_image = cv2.resize(roi, (36, 36), interpolation=cv2.INTER_CUBIC)
            raw_sig.append(downsampled_image)

        elif framework == 'LiCVPR':
            d1 = abs(selected_coordinates[0][0] - selected_coordinates[6][0])
            d2 = abs(selected_coordinates[1][0] - selected_coordinates[5][0])
            d3 = abs(selected_coordinates[2][0] - selected_coordinates[4][0])

            d4 = abs(selected_coordinates[8][1] - selected_coordinates[2][1])
            d5 = abs(selected_coordinates[7][1] - selected_coordinates[4][1])
            d6 = abs(selected_coordinates[3][1] - selected_coordinates[6][1])

            extension = [(int(0.05 * d1), 0), (int(0.075 * d2), 0), (int(0.1 * d3), 0), (0, -int(0.075 * d6)),
                         (-int(0.1 * d3), 0), (-int(0.075 * d2), 0), (-int(0.05 * d1), 0), (0, int(0.3 * d4)),
                         (0, int(0.3 * d5))]

            facial_landmark_coordinates = [(x[0] + y[0], x[1] + y[1]) for x, y in zip(selected_coordinates, extension)]

            contour = np.array(facial_landmark_coordinates, dtype=np.int32)
            contour = contour.reshape((-1, 1, 2))

            mask = np.zeros(region.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

            region_of_interest = cv2.bitwise_and(region, region, mask=mask)

        elif framework == 'GREEN':
            if ROI_type == 'ROI_I':
                x1 = selected_coordinates[0][0]
                y1 = selected_coordinates[4][1]
                x2 = selected_coordinates[3][0]
                distance = abs(selected_coordinates[1][1] - selected_coordinates[2][1])
                y2 = int(selected_coordinates[1][1] + distance * 0.1)

                roi = frame[y1:y2, x1:x2]

            elif ROI_type == 'ROI_II':
                x1 = selected_coordinates[0][0]
                y1 = selected_coordinates[4][1]
                x2 = selected_coordinates[3][0]
                distance = abs(selected_coordinates[1][1] - selected_coordinates[2][1])
                y2 = int(selected_coordinates[1][1] + distance * 0.1)

                x = int((x1 + x2) / 2)
                y = int(abs(y2 - y1) * 0.3 + y1)

                roi = frame[y, x]

            elif ROI_type == 'ROI_III':
                x1_new = int(x2 - (abs(x1 - x2) * 0.1))
                y1_new = y
                x2_new = int(x2 + (abs(x1 - x2) * 0.2))
                y2_new = y + int(h / 2)

                roi = frame[y1_new:y2_new, x1_new:x2_new]

            elif ROI_type == 'ROI_IV':
                h, w, _ = frame.shape
                x1 = int(w * 0.01)
                y1 = int(h * 0.06)
                x2 = int(w * 0.96)
                y2 = int(h * 0.98)
                roi = frame[y1:y2, x1:x2]

            else:
                assert False, "Invalid ROI type for the 'GREEN' framework. Please choose one of the valid ROI " \
                              "types: 'ROI_I', 'ROI_II', 'ROI_III', or 'ROI_IV' "

            region_of_interest = roi

        else:
            assert False, "Invalid framework. Please choose one of the valid available frameworks " \
                          "types: 'PCA', 'CHROM', 'ICA', 'LiCVPR', or 'GREEN' "

        if ROI_type != 'ROI_II':
            if pixel_filtering:
                filtered_roi = simple_skin_selection(region_of_interest, lower_rgb=75, higher_rgb=200)
            else:
                filtered_roi = region_of_interest

            if mask is not None and framework == 'LiCVPR':
                b, g, r, a = cv2.mean(filtered_roi, mask=mask)
            else:
                b, g, r = calculate_mean_rgb(filtered_roi)
        else:
            b, g, r = region_of_interest

        raw_sig.append([r, g, b])

    return raw_sig


def extract_raw_bg_signal(input_video, color='g'):
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

    # model_path = 'Necessary_Files\\selfie_segmenter_landscape.tflite'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'remote_PPG', 'Necessary_Files', 'selfie_segmenter_landscape.tflite')
    # model_path = r'C:\Users\ilyas\PycharmProjects\pythonProject1\remote_PPG\Necessary_Files\selfie_segmenter_landscape.tflite'

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create an image segmenter instance with the video mode:
    options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path),
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

