#!/usr/bin/env python3
import torch
from datasets.flying_chairs import FlyingChairsDataset
from torchvision import transforms
from models.FlowNetC import FlowNetC
from torch.optim.lr_scheduler import LinearLR
#from image_lib.core import display_flow_tensor
from models.util import MultiScale_EPE_Loss, EPE_Loss
import time

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

# Hyperparameters
lr = 1e-4
b = (0.9, 0.999)

# Parameters for the dataloader
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 4}
epochs = 150000
running_multi_loss = 0.0
running_final_loss = 0.0
transforms = transforms.Compose([transforms.ToTensor()])

# Datasets and Loaders
dataset = FlyingChairsDataset('datasets/FlyingChairs_release/data', transforms)
set_size = dataset.__len__()
train_size = int(0.9 * set_size)
test_size = set_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
test_loader = torch.utils.data.DataLoader(test_dataset, **params)

# Model
flowNet = FlowNetC(device=device).to(device)

# Optimizers
optim = torch.optim.Adam(flowNet.parameters(), lr=lr, betas=b, weight_decay=4e-4)
#scheduler = LinearLR(optim, start_factor=0.01, total_iters=10000)

training_iterations = 0
update = 1000
stop = False

while True:
    for im1, im2, flow in train_loader:
        if use_cuda:
            im1 = im1.to(device)
            im2 = im2.to(device)
            flow = flow.to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pred_flow = flowNet(im1, im2)
        loss = MultiScale_EPE_Loss(pred_flow, flow)
        final_loss = EPE_Loss(pred_flow[0], flow)

        #update weights
        optim.zero_grad()
        loss.backward()
        optim.step()

        #update running loss
        running_multi_loss += loss.item()
        running_final_loss += final_loss.item()
        #scheduler.step()
        end.record()

        torch.cuda.synchronize()
        if training_iterations >= 100000 and training_iterations % 25000 == 0:
            optim.state_dict()["lr"] *= 0.5

        if training_iterations % update == 0:
            print(f'[Iterations|Multi-EPE-loss|Final-EPE-loss|Runtime]: {training_iterations + 1} | {running_multi_loss / update :.3f} | {running_final_loss / update :.3f} | {start.elapsed_time(end) : .1f}')
            running_multi_loss = 0.0
            running_final_loss = 0.0
        training_iterations += 1
        if training_iterations % 50000:
            torch.save(flowNet.state_dict(), 'model.pt')
        if training_iterations >= epochs:
            stop = True
            break
    if stop:
        break

torch.save(flowNet.state_dict(), 'model.pt')
