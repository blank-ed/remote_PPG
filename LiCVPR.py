# 1) VJ to detect face (DONE)
# 2) DRMF method to find the coordinates of 66 facial landmarks
# 3) Using l=9 points out of 66 to mark and define the ROI
# 4) feature points are detected inside the rectangle using standard 'good feature to track' and are tracked through the following frames using KLT algorithm
# 5) get the raw green value (DONE)
# 6) Illumination rectification (DONE)
# 7) Non-rigid motion elimination (DONE)
# 8) 3 filters: detrending filter, moving-average filter, hamming window based finite impulse response filter with cutoff frequency of 0.7 to 4 hz (DONE)
# 9) after filtering, PSD with welchs method is applied (DONE)


"""

This module contains the framework implemented by
https://openaccess.thecvf.com/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf
also known as LiCVPR rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from remote_PPG.utils import *
from remote_PPG.filters import *
from scipy.signal import welch
import numpy as np
from scipy.stats import cumfreq


def licvpr_framework(input_video):
    """
    :param input_video:
        This takes in an input video file
    :return:
        Returns the estimated heart rate of the input video based on LiCVPR framework
    """

    raw_green_sig = vj_face_detector(input_video, framework='LiCVPR', width=1, height=1)  # Get the raw green signal
    fps = get_fps(input_video)  # find the fps of the video
    raw_bg_green_signal = raw_bg_signal(input_video, color='g')  # Get the raw background green signal

    # Apply the Illumination Rectification filter
    g_ir = rectify_illumination(face_color=np.array(raw_green_sig), bg_color=np.array(raw_bg_green_signal))

    motion_eliminated = non_rigid_motion_elimination(signal=g_ir.tolist(), segment_length=1, fps=fps, threshold=0.05)

    detrended = detrending_filter(signal=np.array(motion_eliminated), Lambda=300)
    moving_average = moving_average_filter(signal=detrended, window_size=3)
    bp_filtered = fir_bp_filter(moving_average, fps=30, low=0.7, high=4)

    frequencies, psd = welch(bp_filtered, fs=30, nperseg=256, nfft=2048)

    first = np.where(frequencies > 0.7)[0]
    last = np.where(frequencies < 4)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)
    max_idx = np.argmax(psd[range_of_interest])
    f_max = frequencies[range_of_interest[max_idx]]
    hr = f_max * 60.0

    return hr


def rectify_illumination(face_color, bg_color, step=0.003, length=3):
    """performs illumination rectification.

    The correction is made on the face green values using the background green values,
    to remove global illumination variations in the face green color signal.

    Parameters
    ----------
    face_color: numpy.ndarray
      The mean green value of the face across the video sequence.
    bg_color: numpy.ndarray
      The mean green value of the background across the video sequence.
    step: float
      Step size in the filter's weight adaptation.
    length: int
      Length of the filter.

    Returns
    -------
    rectified color: numpy.ndarray
      The mean green values of the face, corrected for illumination variations.

    """
    # first pass to find the filter coefficients
    # - y: filtered signal
    # - e: error (aka difference between face and background)
    # - w: filter coefficient(s)
    yg, eg, wg = nlms(bg_color, face_color, length, step)

    # second pass to actually filter the signal, using previous weights as initial conditions
    # the second pass just filters the signal and does NOT update the weights !
    yg2, eg2, wg2 = nlms(bg_color, face_color, length, step, initCoeffs=wg, adapt=False)
    return eg2


def nlms(signal, desired_signal, n_filter_taps, step, initCoeffs=None, adapt=True):
    """Normalized least mean square filter.

    Based on adaptfilt 0.2:  https://pypi.python.org/pypi/adaptfilt/0.2

    Parameters
    ----------
    signal: numpy.ndarray
      The signal to be filtered.
    desired_signal: numpy.ndarray
      The target signal.
    n_filter_taps: int
      The number of filter taps (related to the filter order).
    step: float
      Adaptation step for the filter weights.
    initCoeffs: numpy.ndarray
      Initial values for the weights. Defaults to zero.
    adapt: bool
      If True, adapt the filter weights. If False, only filters.

    Returns
    -------
    y: numpy.ndarray
      The filtered signal.

    e: numpy.ndarray
      The error signal (difference between filtered and desired)

    w: numpy.ndarray
      The found weights of the filter.

    """
    eps = 0.001
    number_of_iterations = len(signal) - n_filter_taps + 1
    if initCoeffs is None:
        initCoeffs = np.zeros(n_filter_taps)

    # Initialization
    y = np.zeros(number_of_iterations)  # Filter output
    e = np.zeros(number_of_iterations)  # Error signal
    w = initCoeffs  # Initial filter coeffs

    # Perform filtering
    errors = []
    for n in range(number_of_iterations):
        x = np.flipud(signal[n:(n + n_filter_taps)])  # Slice to get view of M latest datapoints
        y[n] = np.dot(x, w)
        e[n] = desired_signal[n + n_filter_taps - 1] - y[n]
        errors.append(e[n])

        if adapt:
            normFactor = 1. / (np.dot(x, x) + eps)
            w = w + step * normFactor * x * e[n]
            y[n] = np.dot(x, w)

    return y, e, w


def non_rigid_motion_elimination(signal, segment_length, fps, threshold=0.05):
    """
    :param signal:
        Input signal to segment
    :param segment_length:
        The length of each segment in seconds (s)
    :param fps:
        The frame rate of the video
    :param threshold:
        The cutoff threshold of the segments based on their standard deviation
    :return:
        Returns motion eliminated signal
    """

    # Divide the signal into m segments of the same length
    segments = []
    for i in range(0, len(signal), int(segment_length * fps)):
        end = i + int(segment_length * fps)
        if end > len(signal):
            end_segment_index = i
            break
        segments.append(signal[i:end])
    else:
        end_segment_index = len(segments) * fps

    sd = np.array([np.std(segment) for segment in segments])  # Find the standard deviation of each segment

    # calculate the cumulative frequency of the data, which is effectively the CDF
    # 'numbins' should be set to the number of unique standard deviations
    a = cumfreq(sd, numbins=len(np.unique(sd)))

    # get the value that is the cut-off for the top 5% which is done by finding the smallest standard deviation that
    # has a cumulative frequency greater than 95% of the data
    cut_off_index = np.argmax(a.cumcount >= len(sd) * (1 - threshold))
    cut_off_value = a.lowerlimit + np.linspace(0, a.binsize * a.cumcount.size, a.cumcount.size)[cut_off_index]

    # create a mask where True indicates the value is less than the cut-off
    mask = sd < cut_off_value

    # get the new list of segments excluding the top 5% of highest SD
    segments_95_percent = np.concatenate((np.array(segments)[mask]), axis=None)

    # Add residual signal (leftover original signal) due to segmentation if there is any
    if len(signal) != end_segment_index:
        residual_signal = np.array(signal[end_segment_index:len(signal)])
        motion_eliminated = np.concatenate((segments_95_percent, residual_signal), axis=None)
    else:
        motion_eliminated = segments_95_percent

    return motion_eliminated

