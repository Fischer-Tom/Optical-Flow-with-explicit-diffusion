#!/usr/bin/env python3
import torch
from datasets.flying_chairs import FlyingChairsDataset
from torchvision import transforms
from models.FlowNetC import FlowNetC
from torch.optim.lr_scheduler import LinearLR
#from image_lib.core import display_flow_tensor
import time

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

# Hyperparameters
lr = 1e-4
b = (0.9, 0.999)

# Parameters for the dataloader
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 4}
epochs = 600000
running_loss = 0.0
transforms = transforms.Compose([transforms.ToTensor()])

# Datasets and Loaders
dataset = FlyingChairsDataset('datasets/FlyingChairs_release/data', transforms)
set_size = dataset.__len__()
train_size = int(0.8 * set_size)
test_size = set_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
test_loader = torch.utils.data.DataLoader(test_dataset, **params)

# Model
flowNet = FlowNetC(device=device).to(device)

# Optimizers
optim = torch.optim.Adam(flowNet.parameters(), lr=lr, betas=b, weight_decay=4e-4)
scheduler = LinearLR(optim, start_factor=0.01, total_iters=10000)

training_iterations = 0
update = 200
stop = False

while True:
    for im1, im2, flow in train_loader:
        if use_cuda:
            im1 = im1.to(device)
            im2 = im2.to(device)
            flow = flow.to(device)
        optim.zero_grad()
        pred_flow = flowNet(im1, im2)
        loss = torch.norm(flow - pred_flow, p=2, dim=1).mean()
        loss.backward()
        optim.step()
        running_loss += loss.item()
        scheduler.step()
        if training_iterations >= 300000 and training_iterations % 100000 == 0:
            optim.state_dict()["lr"] *= 0.5

        training_iterations += 1

        if training_iterations % update == 0:
            print(f'[{training_iterations + 1}]: EPE-loss: {running_loss / update :.3f}')
            running_loss = 0.0
        if training_iterations % 100000:
            torch.save(flowNet.state_dict(), 'model.pt')
            break
        if training_iterations >= epochs:
            stop = True
    if stop:
        break

torch.save(flowNet.state_dict(), 'model.pt')
