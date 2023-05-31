import numpy as np
from scipy.signal import firwin, filtfilt, medfilt, butter
import cv2


def normalize(signal):
    """
    :param signal:
        Input signal to normalize to have zero-mean and unit variance
    :return:
        Normalized signal in [[R] [G] [B]] format
    """
    signal = np.array(signal)
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)
    normalized_signal = (signal - mean) / std_dev

    # Turn normalized signal to [[R] [G] [B]]
    normalized = np.array([normalized_signal[:, i] for i in range(0, 3)])
    return normalized


def fir_bp_filter(signal, fps, low=0.5, high=3.7):
    """
    :param signal:
        Takes in the signal to be bandpass filtered
    :param fps:
        This is the fps of the video file, which is also the sampling frequency
    :param low:
        This is the low frequency level
    :param high:
        This is the high frequency level
    :return:
        Returns the bandpass filtered signal in [[R] [G] [B]] format
    """
    signal = np.array(signal)

    # Coefficients of FIR bandpass filter
    filter_coefficients = firwin(numtaps=32, cutoff=[low, high], fs=fps, pass_zero=False, window='hamming')

    # Filtering using the FIR bandpass filter coefficients.
    # Since its FIR bandpass filter, the denominator coefficient is set as 1
    filtered_signal = filtfilt(filter_coefficients, 1, signal, axis=0)

    return filtered_signal


def butterworth_bp_filter(signal, fps, low=0.5, high=3.7, order=4):
    """
    :param signal:
        Takes in the signal to be bandpass filtered using butterworth
    :param fps:
        This is the fps of the video file, which is also the sampling frequency
    :param low:
        This is the low frequency level
    :param high:
        This is the high frequency level
    :param order:
        Filter order
    :return:
        Returns the bandpass filtered signal
    """

    signal = np.array(signal)

    # Coefficients of butterworth bandpass filter
    b, a = butter(order, Wn=[low, high], fs=fps, btype='bandpass')

    # Filtering using the butterworth bandpass filter coefficients
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def detrending_filter(signal, Lambda=300):
    """
    This code is based on the following article "An advanced detrending method with application to HRV analysis".
    Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002 and is taken from bob.rppg.base

    :param signal: numpy.ndarray
        The signal where you want to remove the trend.
    :param Lambda: int
        The smoothing parameter.
    :return filtered_signal: numpy.ndarray
        The detrended signal.
    """

    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix
    from scipy.sparse import spdiags
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def moving_average_filter(signal, window_size):
    """
    :param signal:
         Takes in the signal to perform moving average filter on
    :param window_size:
        Window size to perform moving average (number of frames)
    :return:
        Returns moving average filtered signal
    """

    moving_averages = []
    for i in range(len(signal) - window_size + 1):
        moving_averages.append(sum(signal[i:i+window_size])/window_size)

    return moving_averages


def simple_skin_selection(frame, lower_rgb=75, higher_rgb=200):
    """
    :param frame:
        Input frames of video
    :param lower_rgb:
        Lower RGB threshold level
    :param higher_rgb:
        Higher RGB threshold level
    :return:
        Returns filtered pixels that lies between given RGB threshold
    """
    lower_rgb_threshold = np.any(frame < lower_rgb, axis=-1)  # Lower RGB threshold
    higher_rgb_threshold = np.any(frame > higher_rgb, axis=-1)  # Higher RGB threshold
    indices = np.logical_or(lower_rgb_threshold, higher_rgb_threshold) # Combine these indices

    img_copy = frame.copy()  # Create a copy of the image to not overwrite the original one
    img_copy[indices] = 0  # Change these pixels to black

    return img_copy


def hsv_skin_selection(frame, alpha=0.2, filter_length=5):
    """
    This HSV skin selection algorithm is based on Lee, H., Ko, H., Chung, H., Nam, Y., Hong, S. and Lee, J., 2022.
    Real-time realizable mobile imaging photoplethysmography. Scientific Reports, 12(1), p.7141 which is available at
    https://www.nature.com/articles/s41598-022-11265-x

    :param frame:
        Input frames of video
    :param alpha:
        Constant alpha used to adjust the skin extraction regions
    :param filter_length:
        Median filter length
    :return:
        Returns filtered skin pixels
    """


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    h, s, v = cv2.split(hsv)  # Split HSV image into H, S and V channels
    histogram = cv2.calcHist([s], [0], None, [256], [0, 256])  # Calculate histogram of S channel
    saturation = [x[0] for x in histogram]

    filtered_data = medfilt(saturation, kernel_size=filter_length).tolist()  # Filter using median filter of length 5
    hist_max = filtered_data.index(max(filtered_data))  # Find the frequent saturation value

    # Calculate the threshold range based on alpha and maximum/frequent saturation value
    TH_range = alpha * hist_max
    TH_max = hist_max + TH_range / 2.0
    TH_min = hist_max - TH_range / 2.0

    mask = cv2.inRange(s, TH_min, TH_max)  # create a boolean mask where saturation value is between TH_min and TH_max

    selected_pixels = cv2.bitwise_and(frame, frame, mask=mask)  # apply the mask to the original image

    return selected_pixels
