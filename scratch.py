#!/usr/bin/env python3
import torch
from datasets.flying_chairs import FlyingChairsDataset
from torchvision import transforms
from models.FlowNetC import FlowNetC
from torch.optim.lr_scheduler import StepLR
from image_lib.core import display_flow_tensor
import time


use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

# Hyperparameters
lr = 1e-6
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
flowNet = FlowNetC(device=device)
flowNet = torch.nn.DataParallel(flowNet).to(device)

# Optimizers
optim = torch.optim.Adam(flowNet.parameters(), lr=lr, betas=b)
lr_helper = 1e1
#scheduler = StepLR(optim, step_size=100, gamma=lr_helper**(1./5.))

training_iterations = 0
update = 100

im1, im2, flow = next(iter(train_loader))
for epoch in range(epochs):

    #for im1, im2, flow in train_loader:
    if use_cuda:
        im1 = im1.to(device)
        im2 = im2.to(device)
        flow = flow.to(device)
    optim.zero_grad()
    pred_flow = flowNet(im1, im2)
    loss = torch.norm(flow-pred_flow, p=2, dim=1).mean()
    loss.backward()
    optim.step()
    running_loss += loss.item()
#    scheduler.step()
#    if training_iterations >= 10000:
#        scheduler.state_dict()["step_size"] = 10000
#        scheduler.state_dict()["gamma"] = 0.5

    training_iterations += 1

    if training_iterations % update == 0:
        print(f'[{epoch+1}, {training_iterations+1}]: EPE-loss: {running_loss/update :.3f}')
        running_loss = 0.0
    if training_iterations > epochs:
        torch.save(flowNet.state_dict(), 'model.pt')
        break

#display_flow_tensor(pred_flow[0].detach())
torch.save(flowNet.state_dict(), 'model.pt')
