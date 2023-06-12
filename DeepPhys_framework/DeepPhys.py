# RGB Video 1 was center cropped at 492x492 pixels
# RGB Video 2 and MAHNOB-HCI videos -> detect face and a square region of 160% width and height of the bb was cropped
# input of motion representation is downsampling each frame to 36x36 pixels squared using bicubic interpolation (DONE)

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adadelta
from torch.nn import MSELoss
from DeepPhys_model import DeepPhys
from remote_PPG.utils import *
from scipy.interpolate import PchipInterpolator


def piecewise_cubic_hermite_interpolator(ground_truth_signal, sampling_rate, fps):
    # Your original data
    ground_truth = np.array(ground_truth_signal)

    # Original time vector (sampling_rate Hz)
    t_original = np.arange(0, len(ground_truth) / sampling_rate, 1 / sampling_rate)

    # Interpolation time vector (fps Hz)
    t_interpolated = np.arange(0, len(ground_truth) / sampling_rate, 1 / fps)

    # Create a PCHIP interpolator object
    pchip_interpolator = PchipInterpolator(t_original, ground_truth)

    # Use the interpolator object to interpolate
    interpolated_data = pchip_interpolator(t_interpolated)

    return interpolated_data


class DatasetDeepPhys(Dataset):
    """
        Dataset class for training network.
    """

    def __init__(self, input_video, ground_truth):
        """

        :param path: Path to hdf5 file
        :param labels: tuple of label names to use (e.g.: ('pulseNumerical', 'resp_signal') or ('pulse_signal', ) )
            Note that the first label must be the pulse rate if it is present!
        """

        self.input_video = input_video

        frames = extract_raw_sig(self.input_video, framework='DeepPhys', width=1, height=1)

        # Process ground truth data
        pt = []
        for i in range(0, len(ground_truth) - 1):
            pt.append(ground_truth[i + 1] - ground_truth[i])

        # Scaled to unit standard deviation over each video for training label
        mean_tr_la = np.mean(pt)
        std_dev_tr_la = np.std(pt)
        normalized_tr_la = (np.array(pt) - mean_tr_la) / std_dev_tr_la
        pchip_data = piecewise_cubic_hermite_interpolator(normalized_tr_la, 60, 30)

        self.label = pchip_data[0:len(frames)-1]

        # Process the input video
        std = np.std(frames)
        dlt = []
        torch_frames = []

        for i in range(0, len(frames) - 1):
            img1 = frames[i]
            img2 = frames[i+1]

            img1 = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1)
            torch_frames.append(img1)
            img2 = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1)

            normalized_difference = torch.div(img2 - img1 + 1e-8, img1 + img2)
            clipped = torch.clamp(normalized_difference, -3 * std, 3 * std)
            dlt.append(clipped)

        dlt = torch.stack(dlt, dim=0)
        torch_frames = torch.stack(torch_frames, dim=0)

        # Scaled to unit standard deviation over each video for motion model input
        mean_dlt = torch.mean(dlt, dim=0)
        std_dev_dlt = torch.std(dlt, dim=0)
        normalized_dlt = (dlt - mean_dlt) / std_dev_dlt

        # centered to zero mean and scaled to unit standard deviation as input for appearance model
        a_mean = torch.mean(torch_frames, dim=0)
        a_std_dev = torch.std(torch_frames, dim=0)
        a_normalized = (torch_frames - a_mean) / a_std_dev

        self.M = normalized_dlt
        self.A = a_normalized

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct target signals
        target = torch.tensor(self.label[idx]).float()

        # Construct networks input
        A = self.A[idx]
        M = self.M[idx]

        # Video shape: C x D x H X W
        return A, M, target


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

model = DeepPhys().to(device)
loss_function = MSELoss().to(device)
optimizer = Adadelta(model.parameters())

input_video = r"C:\Users\Admin\Desktop\Riccardo New Dataset\test_L00_no_ex_riccardo_all_distances\D01.mp4"
ground_truth = [31, 31, 31, 31, 31, 31, 31, 31, 30, 30, 30, 32, 34, 34, 42, 48, 48, 59, 62, 62, 57, 57, 53, 51, 51, 49, 48, 46, 45, 44, 44, 43, 43, 42, 42, 41, 40, 38, 37, 36, 34, 33, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 33, 33, 33, 33, 32, 32, 32, 32, 31, 31, 31, 31, 30, 30, 30, 30, 30, 31, 31, 32, 31, 32, 34, 37, 42, 48, 54, 62, 63, 62, 59, 56, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 43, 42, 40, 40, 39, 37, 36, 35, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29, 30, 30, 30, 29, 29, 29, 30, 36, 41, 47, 53, 58, 60, 61, 59, 57, 54, 52, 50, 48, 47, 47, 46, 46, 45, 45, 44, 44, 44, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 34, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 31, 31, 30, 30, 30, 31, 31, 31, 30, 30, 30, 30, 30, 30, 29, 29, 30, 31, 34, 38, 43, 49, 54, 59, 61, 60, 58, 55, 53, 50, 49, 47, 46, 45, 45, 44, 44, 45, 45, 45, 45, 45, 44, 43, 41, 40, 39, 38, 38, 38, 37, 37, 37, 37, 37, 36, 35, 35, 34, 34, 34, 32, 32, 32, 32, 31, 31, 31, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 30, 31, 31, 38, 42, 42, 54, 59, 59, 62, 60, 60, 55, 53, 53, 49, 48, 48, 46, 46, 46, 45, 45, 45, 46, 46, 46, 44, 43, 43, 41, 40, 40, 39, 39, 39, 38, 38, 38, 37, 37, 37, 36, 35, 35, 34, 33, 33, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 32, 31, 31, 31, 31, 31, 37, 41, 41, 53, 58, 58, 63, 62, 62, 56, 54, 54, 51, 50, 50, 49, 47, 46, 46, 45, 45, 45, 44, 43, 42, 41, 40, 39, 37, 36, 35, 35, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 30, 30, 31, 32, 35, 40, 45, 52, 58, 63, 65, 65, 63, 60, 58, 56, 55, 54, 52, 51, 50, 48, 47, 46, 46, 45, 45, 44, 44, 43, 42, 40, 40, 39, 38, 37, 37, 37, 37, 37, 37, 37, 37, 37, 36, 36, 35, 34, 34, 33, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 30, 31, 31, 31, 31, 31, 31, 31, 33, 36, 41, 46, 52, 57, 60, 62, 61, 59, 56, 54, 52, 50, 48, 47, 46, 46, 46, 46, 46, 46, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 37, 37, 37, 37, 37, 38, 38, 38, 37, 37, 36, 35, 35, 34, 34, 33, 33, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 32, 32, 33, 36, 41, 46, 53, 58, 62, 64, 63, 61, 59, 56, 54, 53, 51, 50, 49, 48, 47, 47, 46, 47, 47, 47, 47, 46, 46, 44, 43, 42, 42, 40, 40, 40, 39, 39, 39, 38, 38, 37, 37, 37, 36, 35, 35, 34, 33, 33, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 32, 32, 36, 40, 40, 53, 59, 59, 65, 64, 64, 60, 58, 58, 55, 54, 54, 51, 50, 50, 48, 47, 47, 46, 46, 46, 44, 43, 43, 41, 40, 40, 39, 38, 38, 37, 36, 36, 37, 37, 37, 37, 37, 37, 36, 36, 35, 35, 35, 33, 33, 33, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 32, 34, 34, 42, 48, 48, 59, 62, 62, 61, 58, 56, 54, 53, 52, 51, 50, 49, 48, 48, 47, 47, 47, 47, 46, 46, 45, 44, 43, 42, 41, 40, 38, 38, 38, 38, 38, 37, 37, 36, 35, 35, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 33, 33, 36, 40, 45, 51, 57, 61, 63, 63, 61, 59, 57, 55, 53, 52, 50, 49, 48, 47, 47, 47, 47, 48, 47, 47, 45, 44, 43, 42, 41, 40, 40, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38, 38, 38, 38, 37, 36, 36, 35, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 33, 34, 38, 42, 47, 53, 57, 60, 61, 60, 58, 57, 55, 54, 54, 53, 53, 52, 51, 50, 49, 47, 46, 46, 45, 44, 42, 41, 40, 40, 39, 39, 39, 39, 38, 38, 38, 37, 37, 36, 36, 36, 36, 36, 36, 36, 35, 35, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 32, 34, 38, 43, 49, 55, 60, 63, 64, 62, 60, 58, 56, 55, 53, 52, 51, 49, 48, 47, 46, 46, 45, 45, 44, 44, 43, 41, 41, 40, 38, 38, 36, 36, 36, 35, 35, 35, 35, 35, 35, 35, 34, 34, 33, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 40, 45, 45, 55, 58, 58, 60, 58, 58, 54, 52, 52, 48, 47, 47, 46, 46, 46, 46, 46, 46, 46, 46, 44, 43, 43, 40, 40, 40, 39, 39, 39, 37, 37, 37, 36, 35, 35, 34, 33, 33, 32, 32, 32, 31, 31, 31, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 32, 32, 40, 44, 44, 55, 59, 60, 60, 58, 56, 53, 52, 50, 47, 47, 46, 46, 46, 46, 46, 46, 46, 46, 45, 44, 43, 42, 40, 40, 39, 38, 37, 37, 36, 36, 35, 35, 35, 34, 34, 33, 32, 32, 32, 31, 31, 31, 30, 30, 30, 30, 30, 30, 30, 31, 30, 30, 30, 30, 30, 30, 29, 30, 32, 36, 40, 46, 52, 57, 59, 60, 58, 54, 52, 51, 50, 49, 48, 47, 46, 45, 45, 44, 44, 44, 44, 43, 42, 41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 34, 34, 33, 33, 33, 33, 32, 32, 32, 32, 31, 31, 31, 30, 30, 30, 31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 31, 34, 38, 43, 49, 54, 59, 58, 57, 69, 64, 60, 56, 54, 52, 51, 51, 51, 52, 53, 53, 53, 52, 50, 48, 46, 43, 41, 39, 37, 35, 34, 34, 34, 34, 35, 35, 35, 35, 35, 33, 32, 31, 29, 28, 26, 25, 24, 24, 24, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 24, 29, 39, 49, 61, 73, 81, 85, 84, 81, 75, 71, 68, 65, 63, 62, 61, 59, 58, 58, 57, 57, 57, 57, 55, 54, 54, 50, 48, 48, 43, 42, 42, 40, 39, 39, 38, 38, 38, 37, 36, 36, 34, 33, 33, 30, 29, 29, 26, 25, 25, 24, 24, 24, 24, 24, 24, 23, 23, 23, 22, 22, 22, 30, 39, 39, 63, 74, 74, 86, 85, 85, 77, 73, 73, 68, 66, 66, 62, 60, 60, 56, 55, 55, 55, 55, 55, 52, 50, 50, 45, 43, 43, 40, 40, 40, 38, 38, 38, 37, 36, 36, 35, 35, 35, 32, 31, 31, 30, 29, 29, 29, 28, 28, 27, 27, 27, 27, 26, 26, 25, 24, 24, 24, 24, 24, 33, 42, 42, 63, 73, 80, 84, 83, 80, 76, 72, 68, 65, 63, 62, 61, 60, 59, 59, 60, 61, 61, 61, 59, 57, 54, 51, 48, 45, 43, 42, 42, 41, 41, 41, 40, 40, 40, 40, 40, 39, 38, 37, 36, 34, 33, 32, 30, 29, 28, 27, 27, 26, 25, 24, 24, 24, 24, 24, 30, 37, 47, 58, 69, 77, 82, 83, 81, 78, 73, 70, 67, 64, 62, 61, 59, 58, 58, 58, 58, 58, 58, 57, 55, 53, 50, 47, 44, 41, 40, 39, 38, 38, 37, 37, 37, 36, 36, 35, 34, 33, 32, 31, 29, 28, 27, 25, 24, 24, 24, 24, 24, 24, 23, 23, 23, 22, 22, 27, 34, 43, 53, 65, 75, 80, 82, 80, 76, 72, 68, 65, 63, 61, 60, 58, 57, 56, 56, 56, 56, 55, 54, 53, 50, 48, 46, 44, 42, 40, 39, 38, 38, 37, 37, 36, 36, 36, 36, 36, 36, 36, 34, 33, 32, 30, 29, 27, 25, 24, 24, 24, 24, 23, 23, 23, 22, 22, 24, 29, 38, 48, 61, 72, 80, 84, 85, 83, 79, 76, 74, 72, 71, 69, 67, 65, 63, 62, 60, 60, 59, 59, 58, 56, 54, 51, 49, 46, 43, 41, 40, 40, 38, 37, 37, 36, 35, 35, 34, 33, 33, 30, 29, 27, 25, 25, 25, 24, 23, 23, 22, 22, 22, 22, 21, 21, 21, 21, 20, 20, 20, 22, 27, 27, 47, 59, 59, 78, 81, 81, 78, 75, 75, 67, 65, 65, 62, 61, 61, 56, 54, 54, 51, 51, 51, 49, 48, 48, 43, 41, 41, 38, 37, 37, 36, 35, 35, 34, 34, 34, 33, 32, 32, 30, 29, 29, 26, 26, 26, 24, 23, 23, 21, 21, 21, 20, 22, 21, 21, 21, 20, 19, 19, 21, 26, 26, 44, 55, 55, 72, 75, 75, 70, 67, 67, 61, 59, 57, 55, 53, 51, 50, 49, 49, 49, 49, 49, 48, 46, 43, 41, 38, 35, 33, 30, 28, 27, 26, 25, 25, 25, 26, 26, 26, 26, 25, 25, 24, 23, 22, 21, 21, 20, 20, 19, 18, 17, 17, 17, 17, 17, 18, 19, 19, 19, 21, 26, 34, 44, 55, 65, 74, 79, 79, 77, 73, 69, 66, 63, 62, 60, 59, 57, 54, 52, 50, 49, 48, 47, 45, 44, 42, 40, 38, 35, 33, 30, 29, 28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 26, 25, 24, 23, 22, 21, 19, 18, 18, 17, 18, 18, 18, 18, 17, 17, 16, 15, 15, 15, 15, 17, 23, 31, 41, 53, 65, 74, 80, 80, 77, 73, 68, 64, 61, 59, 57, 55, 53, 50, 48, 47, 46, 46, 45, 44, 43, 42, 40, 37, 34, 32, 31, 29, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 21, 20, 19, 18, 18, 18, 17, 16, 15, 15, 14, 14, 15, 15, 15, 15, 16, 15, 14, 14, 16, 21, 30, 40, 51, 61, 68, 71, 70, 67, 63, 58, 54, 50, 48, 46, 45, 44, 43, 43, 44, 45, 45, 45, 44, 42, 40, 37, 34, 34, 31, 29, 29, 27, 27, 27, 25, 25, 25, 25, 25, 25, 24, 24, 22, 21, 21, 20, 20, 20, 19, 19, 19, 17, 16, 16, 16, 16, 16, 17, 20, 20, 31, 41, 41, 64, 74, 74, 81, 78, 78, 69, 65, 65, 59, 57, 57, 53, 51, 51, 48, 48, 48, 47, 45, 45, 41, 39, 39, 33, 30, 30, 27, 26, 26, 26, 26, 26, 26, 26, 26, 25, 25, 24, 24, 24, 23, 23, 23, 21, 21, 21, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 23, 23, 40, 51, 51, 73, 80, 80, 82, 79, 75, 71, 67, 64, 61, 60, 58, 56, 54, 52, 50, 49, 48, 47, 46, 44, 42, 40, 38, 36, 34, 33, 31, 31, 31, 31, 31, 31, 31, 31, 30, 30, 29, 27, 26, 24, 24, 23, 22, 22, 22, 22, 21, 22, 22, 21, 21, 21, 22, 22, 22, 22, 22, 25, 31, 40, 50, 61, 71, 77, 80, 80, 76, 72, 67, 63, 60, 57, 55, 53, 53, 52, 52, 53, 53, 52, 51, 49, 47, 45, 42, 40, 38, 36, 33, 33, 32, 32, 32, 32, 33, 33, 32, 31, 30, 29, 28, 27, 25, 25, 24, 23, 23, 23, 22, 22, 22, 22, 22, 22, 21, 22, 23, 24, 28, 35, 44, 55, 66, 73, 77, 77, 74, 70, 65, 61, 57, 55, 53, 52, 51, 51, 51, 52, 52, 53, 53, 52, 50, 48, 46, 43, 40, 37, 37, 36, 35, 34, 33, 32, 32, 31, 30, 29, 28, 26, 25, 24, 24, 23, 22, 22, 21, 21, 20, 20, 20, 20, 20, 21, 21, 21, 20, 21, 23, 29, 39, 49, 60, 70, 77, 80, 79, 76, 72, 68, 64, 62, 59, 57, 56, 53, 51, 51, 50, 50, 50, 50, 49, 48, 46, 43, 41, 39, 37, 37, 35, 35, 35, 34, 34, 34, 31, 30, 30, 27, 26, 26, 24, 23, 23, 22, 21, 21, 20, 19, 19, 19, 19, 19, 20, 20, 20, 19, 19, 19, 17, 17, 17, 24, 33, 33, 55, 66, 66, 78, 77, 77, 70, 66, 66, 62, 60, 60, 56, 54, 54, 49, 47, 47, 44, 43, 43, 41, 40, 40, 36, 33, 33, 29, 27, 27, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 21, 21, 21, 20, 19, 19, 18, 18, 18, 18, 17, 17, 18, 19, 19, 19, 19, 19, 26, 34, 34, 54, 63, 63, 72, 70, 70, 62, 58, 54, 51, 48, 46, 44, 42, 41, 40, 41, 42, 42, 40, 39, 36, 33, 31, 28, 26, 25, 24, 24, 25, 25, 25, 26, 26, 26, 26, 26, 25, 25, 24, 23, 23, 23, 22, 22, 21, 21, 20, 20, 20, 20, 20, 20, 21, 21, 22, 25, 30, 39, 48, 59, 70, 77, 81, 80, 77, 73, 68, 65, 63, 61, 58, 56, 54, 51, 50, 48, 47, 46, 44, 42, 40, 38, 36, 34, 32, 30, 29, 28, 27, 27, 27, 27, 27, 27, 28, 27, 27, 27, 26, 26, 25, 25, 24, 24, 24, 23, 23, 22, 22, 22, 22, 21, 21, 20, 21, 22, 25, 30, 38, 48, 60, 70, 78, 81, 80, 77, 72, 69, 67, 66, 64, 63, 61, 59, 53, 52, 50, 49, 48, 47, 44, 41, 40, 38, 35, 33, 32, 30, 29, 28, 27, 26, 25, 25, 24, 24, 23, 23, 23, 23, 22, 22, 21, 20, 20, 20, 21, 20, 20, 19, 19, 19, 19, 19, 18, 18, 19, 20, 23, 23, 36, 46, 57, 66, 72, 75, 73, 70, 65, 61, 56, 54, 52, 49, 48, 48, 48, 49, 50, 50, 50, 48, 46, 43, 40, 39, 36, 35, 35, 32, 31, 31, 30, 29, 29, 27, 27, 27, 25, 25, 25, 24, 23, 23, 21, 21, 21, 20, 19, 19, 21, 21, 21, 20, 20, 20, 20, 20, 20, 27, 36, 36, 56, 66, 66, 76, 75, 75, 68, 64, 64, 60, 55, 43, 43, 43, 43, 42, 40, 39, 36, 33, 31, 28, 26, 25, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 22, 21, 21, 21, 20, 20, 19, 18, 18, 17, 17, 16, 17, 17, 18, 18, 18, 18, 18, 20, 25, 34, 44, 56, 66, 74, 78, 78, 75, 71, 66, 61, 58, 55, 54, 53, 51, 50, 48, 47, 46, 46, 45, 44, 43, 40, 39, 37, 34, 32, 30, 28, 27, 26, 25, 25, 24, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 20, 20, 19, 18, 17, 17, 16, 16, 16, 17, 17, 17, 18, 17, 16, 17, 22, 29, 39, 49, 60, 68, 73, 70, 65, 61, 57, 54, 52, 49, 47, 44, 43, 42, 41, 42, 42, 42, 41, 40, 38, 36, 33, 30, 28, 25, 24, 24, 23, 24, 24, 25, 25, 25, 25, 25, 25, 24, 23, 22, 23, 23, 22, 21, 21, 20, 19, 19, 19, 18, 19, 20, 20, 21, 20, 21, 24, 31, 40, 49, 60, 69, 77, 75, 71, 66, 61, 58, 55, 52, 50, 49, 48, 47, 47, 47, 48, 48, 48, 46, 44, 42, 40, 38, 35, 33, 32, 31, 30, 29, 28, 28, 27, 26, 25, 24, 24, 24, 24, 23, 23, 23, 22, 22, 21, 21, 21, 21, 22, 22, 22, 21, 21, 21, 20, 20, 22, 27, 35, 45, 56, 76, 80, 81, 79, 74, 71, 67, 64, 60, 58, 55, 54, 53, 52, 52, 52, 52, 52, 51, 50, 48, 46, 43, 41, 40, 38, 36, 35, 35, 34, 33, 33, 31, 30, 30, 28, 27, 27, 25, 24, 24, 24, 23, 23, 23, 22, 22, 22, 22, 22, 21, 22, 22, 22, 22, 22, 21, 23, 23, 45, 45, 66, 74, 74, 78, 76, 76, 67, 63, 63, 57, 55, 55, 53, 52, 52, 50, 51, 51, 51, 50, 50, 47, 45, 45, 40, 39, 39, 36, 35, 35, 33, 32, 32, 31, 30, 30, 28, 27, 27, 25, 24, 24, 23, 22, 22, 21, 21, 21, 20, 20, 20, 21, 21, 21, 21, 21, 21, 26, 26, 43, 53, 53, 73, 77, 77, 76, 72, 72, 64, 60, 60, 55, 54, 52, 51, 51, 51, 51, 51, 51, 50, 49, 47, 44, 42, 40, 39, 37, 36, 35, 34, 33, 33, 32, 31, 31, 30, 29, 27, 26, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 20, 21, 23, 23, 23, 23, 23, 24, 28, 36, 46, 57, 67, 76, 80, 81, 79, 75, 71, 68, 65, 62, 60, 59, 56, 54, 52, 50, 49, 48, 48, 46, 45, 44, 42, 40, 38, 35, 33, 32, 31, 30, 31, 31, 31, 31, 31, 31, 30, 30, 29, 28, 27, 25, 24, 23, 23, 23, 22, 22, 21, 21, 20, 20, 20, 21, 22, 23, 22, 22, 22, 22, 26, 34, 44, 55, 66, 75, 80, 82, 80, 76, 71, 67, 64, 61, 60, 58, 56, 54, 52, 50, 49, 48, 47, 47, 45, 44, 42, 40, 37, 35, 32, 30, 30, 30, 29, 29, 30, 30, 30, 30, 30, 29, 28, 27, 25, 25, 24, 23, 23, 23, 22, 22, 21, 21, 20, 20, 20, 21, 22, 21, 21, 21, 20, 21, 24, 32, 41, 52, 62, 71, 76, 78, 76, 72, 67, 62, 59, 56, 53, 50, 49, 47, 46, 46, 47, 48, 49, 48, 47, 46, 43, 40, 39, 36, 34, 32, 32, 30, 29, 29, 28, 27, 27, 26, 25, 25, 24, 24, 24, 23, 23, 23, 22, 21, 21, 20, 21, 21, 22, 22, 22, 22, 22, 22, 21, 21, 21, 29, 39, 39, 61, 72, 72, 83, 82, 82, 75, 70, 70, 63, 60, 60, 55, 55, 55, 54, 54, 54, 55, 54, 54, 52, 50, 50, 46, 43, 43, 40, 38, 38, 37, 37, 37, 37, 37, 37, 36, 35, 35, 33, 31, 31, 28, 26, 26, 24, 24, 24, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 23, 23, 30, 39, 39, 62, 74, 74, 88, 88, 88, 80, 76, 73, 71, 69, 66, 64, 61, 59, 57, 55, 55, 54, 53, 52, 50, 48, 46, 44, 42, 40, 39, 38, 37, 36, 35, 34, 33, 32, 32, 33, 33, 32, 32, 32, 32, 31, 30, 29, 27, 26, 25, 25, 25, 24, 25, 24, 24, 24, 24, 23, 23, 25, 29, 38, 48, 59, 71, 79, 84, 84, 81, 77, 72, 68, 65, 62, 60, 58, 56, 55, 54, 54, 55, 55, 55, 54, 52, 50, 47, 45, 43, 41, 40, 39, 38, 37, 37, 36, 36, 35, 35, 35, 35, 35, 35, 34, 33, 32, 30, 29, 27, 26, 25, 25, 25, 25, 25, 25, 24, 24, 24, 25, 29, 36, 45, 56, 67, 77, 83, 84, 82, 79, 74, 70, 67, 64, 61, 59, 57, 55, 53, 53, 52, 52, 51, 50, 49, 46, 44, 41, 39, 37, 35, 34, 33, 32, 31, 30, 31, 31, 31, 31, 31, 31, 31, 30, 29, 28, 27, 25, 24, 23, 23, 22, 22, 22, 22, 22, 22, 21, 20, 21, 22, 24, 29, 37, 47, 59, 70, 79, 84, 83, 80, 75, 71, 68, 66, 65, 63, 62, 60, 58, 56, 54, 53, 53, 52, 50, 49, 48, 46, 43, 41, 39, 37, 36, 36, 33, 32, 31, 30, 30, 28, 27, 27, 25, 24, 24, 23, 22, 22, 21, 21, 21, 20, 20, 20, 21, 21, 21, 21, 20, 20, 19, 19, 19, 22, 27, 27, 47, 58, 58, 76, 79, 79, 74, 70, 70, 64, 62, 62, 59, 57, 57, 53, 52, 52, 52, 53, 53, 52, 51, 51, 46, 44, 44, 41, 39, 39, 38, 38, 37, 37, 37, 36, 35, 35, 33, 32, 32, 29, 28, 28, 25, 25, 25, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 25, 25, 38, 47, 47, 69, 77, 77, 81, 78, 78, 70, 68, 66, 64, 63, 61, 59, 57, 56, 55, 55, 54, 54, 52, 50, 48, 46, 44, 42, 40, 39, 38, 38, 37, 36, 36, 35, 34, 33, 32, 31, 29, 28, 26, 25, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 23, 24, 28, 34, 43, 54, 66, 75, 82, 83, 82, 78, 73, 70, 68, 65, 63, 62, 60, 58, 55, 54, 52, 52, 51, 50, 50, 48, 46, 42, 40, 40, 39, 37, 37, 36, 35, 34, 33, 32, 30, 29, 28, 27, 26, 25, 24, 24, 23, 22, 22, 21, 21, 20, 20, 19, 19, 20, 20, 20, 20, 19, 18, 18, 18, 19, 24, 33, 43, 55, 66, 75, 79, 78, 76, 72, 68, 64, 62, 59, 57, 55, 54, 53, 51, 51, 51, 51, 51, 50, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 31, 30, 29, 27, 26, 25, 24, 24, 23, 22, 21, 20, 19, 18, 18, 18, 19, 19, 19]

dataset = DatasetDeepPhys(input_video=input_video, ground_truth=ground_truth)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Start training
model.train()
num_epochs = 16
for epoch in range(num_epochs):
    for A, M, target in dataloader:
        optimizer.zero_grad()

        A, M, target = A.to(device), M.to(device), target.to(device)

        output = model(A, M)

        loss = loss_function(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


# # Define Mean Absolute Error
# def mae(prediction, target):
#     return torch.mean(torch.abs(prediction - target))
#
#
# # Create test data loader
# # test_dataset = VideoDataset(test_video_folder, test_label_folder)
# # test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
# test_dataset = TensorDataset(video_tensor, bvp_tensor)
# test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
#
# # Evaluation
# model.eval()  # Set the model to evaluation mode
# total_mae = 0
# with torch.no_grad():  # Do not calculate gradients for efficiency
#     for video, bvp in test_data_loader:
#         video = video.cuda()  # if using GPU
#         bvp = bvp.cuda()  # if using GPU
#
#         # Forward pass
#         rPPG, x_visual, x_visual3232, x_visual1616 = model(video)
#
#         # Calculate loss
#         rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
#         bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)
#         total_mae += mae(rPPG, bvp).item()
#
# # Calculate average MAE
# avg_mae = total_mae / len(test_data_loader)
# print('The MAE of the test dataset is: ', avg_mae)