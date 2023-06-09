# RGB Video 1 was center cropped at 492x492 pixels
# RGB Video 2 and MAHNOB-HCI videos -> detect face and a square region of 160% width and height of the bb was cropped
# input of motion representation is downsampling each frame to 36x36 pixels squared using bicubic interpolation (DONE)

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adadelta
from torch.nn import MSELoss
from DeepPhys_model import DeepPhys
from remote_PPG.utils import *

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

model = DeepPhys().to(device)
loss_function = MSELoss().to(device)
optimizer = Adadelta(model.parameters())

input_video = r'C:\Users\Admin\Desktop\Riccardo New Dataset\test_L00_no_ex_riccardo_all_distances\D01.mp4'
frames = extract_raw_sig(input_video, framework='DeepPhys', width=1, height=1)



# frames = np.array(extract_raw_sig(input_video, framework='DeepPhys', width=1, height=1))
#
# video_data = frames[0:training_length]
# ground_truth = np.array(ground_truth[0:training_length])
#
# video_tensor = torch.from_numpy(video_data).permute(3, 0, 1, 2).float().unsqueeze(0)  # Add a batch dimension
# bvp_tensor = torch.from_numpy(ground_truth).float().unsqueeze(0)  # Add a batch dimension
#
# dataset = TensorDataset(video_tensor, bvp_tensor)
# data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
#
# # Start training
# model.train()
# num_epochs = 16
# for epoch in range(num_epochs):
#     for video, bvp in data_loader:
#         optimizer.zero_grad()
#
#         video, bvp = video.to(device), bvp.to(device)
#
#         rPPG, x_visual, x_visual3232, x_visual1616 = model(video)
#         rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
#         BVP_label = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize
#
#         loss = loss_function(rPPG, BVP_label)
#
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#
#
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