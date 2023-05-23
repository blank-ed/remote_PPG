# 1) Pick 500 consecutive images exhibiting the smallest amount of inter frame motion
# 2) Apply VJ face detector to define ROI
# 3) Apply simple skin selection process which produces a skin mask inside the ROI that will remove all pixels containing facial hairs and facial features
# 4) Spatial pooling of RGB
# 5) Normalize R, G and B to obtain Rn, Gn and Bn
# 6) Get Xs = 3Rn - 2Gn and Ys = 1.5Rn + Gn - 1.5Bn
# 7) Bandpass filter Xs and Ys to get Xf and Yf
# 8) Find alpha by dividing standard deviation of Xf and standard deviation of Yf
# 9) Find S = Xf - alpha*Yf
# 10)

import cv2

cap = cv2.VideoCapture(r"C:\Users\ilyas\Desktop\VHR\Datasets\Distance vs Light Dataset\test_all_riccardo_distances_L00_NoEx\D01.mp4")
video_sequence = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    video_sequence.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Total frames in the video sequence:", len(video_sequence))

import cv2
import numpy as np

def calculate_intensity(image):
    # calculate the intensity of the image
    return np.sum(image, axis=2) / 3.0

def calculate_motion(images):
    # calculate the motion between consecutive images
    motion = []
    for i in range(0, len(images)):
        motion.append(np.sum(np.abs(calculate_intensity(images[i]) - calculate_intensity(images[i+1]))))
    return motion

def find_least_motion_segment(motion, segment_length):
    # find the segment with the least motion
    cumulative_motion = np.cumsum(motion)
    min_motion = np.inf
    min_index = -1
    for i in range(segment_length, len(cumulative_motion)):
        motion_in_segment = cumulative_motion[i] - cumulative_motion[i-segment_length]
        if motion_in_segment < min_motion:
            min_motion = motion_in_segment
            min_index = i - segment_length + 1
    return min_index

# calculate motion
motion = calculate_motion(video_sequence)

print(motion)
print(len(motion))

# # find the segment with the least motion
# segment_length = 500
# i_s = find_least_motion_segment(motion, segment_length)
#
# print(f'The segment of {segment_length} consecutive frames starting from frame {i_s} has the least motion.')
