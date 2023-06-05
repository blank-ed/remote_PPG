# TODO: Have to implement training model on a specified video dataset with their corresponding ground truth data

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from PhysNetED_BMVC import PhysNet_padding_Encoder_Decoder_MAX
from NegPearsonLoss import Neg_Pearson
import cv2
import numpy as np
from remote_PPG.utils import *


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
