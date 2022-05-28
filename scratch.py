import torch
import matplotlib.pyplot as plt
from datasets.flying_chairs import FlyingChairsDataset
from torchvision import transforms
from image_lib.core import display_flow_tensor
from models.FlowNetS import FlowNetS
from models.FlowNetC import FlowNetC
import time

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Hyperparameters
lr = 1e-4
b = (0.9, 0.999)

# Parameters for the dataloader
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 4}
epochs = 1
running_loss = 0.0
transforms = transforms.Compose([transforms.ToTensor()])

# Datasets and Loaders
train_set = FlyingChairsDataset('datasets/FlyingChairs_release/data', transforms)
train_loader = torch.utils.data.DataLoader(train_set, **params)

# Model
flowNet = FlowNetC(device=device).to(device)

# Optimizers
optim = torch.optim.Adam(flowNet.parameters(), lr=lr, betas=b)

im1, im2, flow = next(iter(train_loader))

for epoch in range(epochs):

    #for im1, im2, flow in train_loader:
    if use_cuda:
        im1 = im1.to(device)
        im2 = im2.to(device)
        flow = flow.to(device)
    optim.zero_grad()
    start = time.time()
    pred_flow = flowNet(im1, im2)
    loss = torch.norm(flow-pred_flow, p=2, dim=1).mean()
    loss.backward()
    end = time.time()
    print(f"Dein Ansatz: {end - start}")
    optim.step()
    running_loss += loss.item()
    if epoch % 500 == 499:
        print(f'[{epoch+1}, {epoch+1}]: EPE-loss: {running_loss / 500:.3f}')
        running_loss = 0.0


display_flow_tensor(flow[0].cpu())
display_flow_tensor(pred_flow[0].detach().cpu())
display_flow_tensor(flow[1].cpu())
display_flow_tensor(pred_flow[1].detach().cpu())
