# TODO: Have to implement training model on a specified video dataset with their corresponding ground truth data

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torch.optim import Adam
from PhysNetED_BMVC import PhysNet_padding_Encoder_Decoder_MAX
from NegPearsonLoss import Neg_Pearson
from remote_PPG.utils import *
import pandas as pd
import matplotlib.pyplot as plt


class PhysNetDatasetBuilder(Dataset):
    def __init__(self, root_dir, dataset=None, training_length=128):

        self.root_dir = root_dir
        self.training_length = training_length
        self.dataset = dataset

        self.video_files = []
        self.gt_files = []

        if self.dataset == 'UBFC1':
            for each_subject in os.listdir(self.root_dir):
                for ground_truth_files in os.listdir(os.path.join(self.root_dir, each_subject)):
                    if ground_truth_files.endswith('.avi'):
                        self.video_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))
                    elif ground_truth_files.endswith('.xmp'):
                        self.gt_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))

        elif self.dataset == 'UBFC2':
            for each_subject in os.listdir(self.root_dir):
                for ground_truth_files in os.listdir(os.path.join(self.root_dir, each_subject)):
                    if ground_truth_files.endswith('.avi'):
                        self.video_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))
                    elif ground_truth_files.endswith('.txt'):
                        self.gt_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))

        # elif self.dataset == None:


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        print(f'Processing: {video_file}')
        frames = np.array(extract_raw_sig(video_file, framework='PhysNet', width=1, height=1))

        gt_file = self.gt_files[idx]
        if self.dataset == 'UBFC1':
            gtdata = pd.read_csv(gt_file, header=None)
            gtTrace = gtdata.iloc[:, 3].tolist()
            gtTrace = gtTrace[::2]  # Resample it to be 30 Hz since the ground truth sensor is at 60 Hz sampling rate

            gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
            gtHR = gtdata.iloc[:, 1]


        elif self.dataset == 'UBFC2':
            gtdata = pd.read_csv(gt_file, delimiter='\t', header=None)
            gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']  # Already resampled
            gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
            gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

        video_data = frames[:self.training_length]
        ground_truth = np.array(gtTrace[0:self.training_length])

        video_tensor = torch.from_numpy(video_data).permute(3, 0, 1, 2).float()  # Add a batch dimension
        bvp_tensor = torch.from_numpy(ground_truth).float()  # Add a batch dimension

        return video_tensor, bvp_tensor




if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(device)
loss_function = Neg_Pearson().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

dataset = PhysNetDatasetBuilder(r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\test', dataset='UBFC1')

# Split dataset
train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Start training
model.train()
num_epochs = 15
for epoch in range(num_epochs):
    for video, bvp in train_loader:
        optimizer.zero_grad()

        video, bvp = video.to(device), bvp.to(device)

        rPPG, x_visual, x_visual3232, x_visual1616 = model(video)
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize

        loss = loss_function(rPPG, BVP_label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


# # Define Mean Absolute Error
# def mae(prediction, target):
#     return torch.mean(torch.abs(prediction - target))
#
# # Create test data loader
# # test_dataset = VideoDataset(test_video_folder, test_label_folder)
# # test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
# test_dataset = TensorDataset(video_tensor, bvp_tensor)
# test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
#
# Evaluation
model.eval()  # Set the model to evaluation mode
total_mae = 0
with torch.no_grad():  # Do not calculate gradients for efficiency
    for video, bvp in test_loader:
        video = video.cuda()  # if using GPU
        bvp = bvp.cuda()  # if using GPU

        # Forward pass
        rPPG, x_visual, x_visual3232, x_visual1616 = model(video)

        # Calculate loss
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
        bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)

        rPPG_filtered = fir_bp_filter(signal=rPPG, fps=30, low=0.67, high=2.0)
        bvp_filtered = fir_bp_filter(signal=bvp, fps=30, low=0.67, high=2.0)

        plt.plot(rPPG_filtered)
        plt.plot(bvp_filtered)
        plt.show()




#         total_mae += mae(rPPG, bvp).item()
#
# # Calculate average MAE
# avg_mae = total_mae / len(test_data_loader)
# print('The MAE of the test dataset is: ', avg_mae)





def physnet_framework(input_video, ground_truth, training_length=128):
    """
    :param input_video:
    :param ground_truth:
    :param training_length:
    :return:
    """

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=training_length).to(device)
    loss_function = Neg_Pearson().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    frames = np.array(extract_raw_sig(input_video, framework='PhysNet', width=1, height=1))

    video_data = frames[0:training_length]
    ground_truth = np.array(ground_truth[0:training_length])

    video_tensor = torch.from_numpy(video_data).permute(3, 0, 1, 2).float().unsqueeze(0)  # Add a batch dimension
    bvp_tensor = torch.from_numpy(ground_truth).float().unsqueeze(0)  # Add a batch dimension

    dataset = TensorDataset(video_tensor, bvp_tensor)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Start training
    model.train()
    num_epochs = 15
    for epoch in range(num_epochs):
        for video, bvp in data_loader:
            optimizer.zero_grad()

            video, bvp = video.to(device), bvp.to(device)

            rPPG, x_visual, x_visual3232, x_visual1616 = model(video)
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
            BVP_label = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize

            loss = loss_function(rPPG, BVP_label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


    # Define Mean Absolute Error
    def mae(prediction, target):
        return torch.mean(torch.abs(prediction - target))

    # Create test data loader
    # test_dataset = VideoDataset(test_video_folder, test_label_folder)
    # test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    test_dataset = TensorDataset(video_tensor, bvp_tensor)
    test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Evaluation
    model.eval()  # Set the model to evaluation mode
    total_mae = 0
    with torch.no_grad():  # Do not calculate gradients for efficiency
        for video, bvp in test_data_loader:
            video = video.cuda()  # if using GPU
            bvp = bvp.cuda()  # if using GPU

            # Forward pass
            rPPG, x_visual, x_visual3232, x_visual1616 = model(video)

            # Calculate loss
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
            bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)
            total_mae += mae(rPPG, bvp).item()

    # Calculate average MAE
    avg_mae = total_mae / len(test_data_loader)
    print('The MAE of the test dataset is: ', avg_mae)
