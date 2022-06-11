#!/usr/bin/env python3
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
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
dataset = FlyingChairsDataset('/home/s8tmfisc/FlyingChairs_release/data', transforms)
set_size = dataset.__len__()
train_size = int(0.8 * set_size)
test_size = int(0.1 * set_size)
val_size = set_size - train_size - test_size
assert (train_size+test_size+val_size <= set_size)
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size],
                                                            generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
test_loader = torch.utils.data.DataLoader(test_dataset, **params)
val_loader = torch.utils.data.DataLoader(val_dataset, **params)

# Model
flowNet = FlowNetC(device=device).to(device)

# Optimizers
optim = torch.optim.Adam(flowNet.parameters(), lr=lr, betas=b, weight_decay=4e-4)
#scheduler = LinearLR(optim, start_factor=0.01, total_iters=10000)


# SummaryWriters
train_writer = SummaryWriter('/home/s8tmfisc/train')
test_writer = SummaryWriter('/home/s8tmfisc/test')

training_iterations = 0
update = 1000
stop = False
min_val_loss = sys.float_info.max

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
            train_writer.add_scalar('mean EPE', running_final_loss / update, training_iterations+1)
            running_multi_loss = 0.0
            running_final_loss = 0.0

        training_iterations += 1

        if training_iterations >= epochs:
            stop = True
            break

    with torch.no_grad():
        val_loss = 0.0
        multiscale_loss = 0.0
        for im1, im2, flow in val_loader:
            if use_cuda:
                im1 = im1.to(device)
                im2 = im2.to(device)
                flow = flow.to(device)
            pred_flow = flowNet(im1, im2)
            loss = MultiScale_EPE_Loss(pred_flow, flow)
            final_loss = EPE_Loss(pred_flow[0], flow)
            multiscale_loss += loss.item()
            val_loss += final_loss.item()
        if val_loss / val_size < min_val_loss:
            min_val_loss = val_loss / val_size
            torch.save(flowNet.state_dict(), '/home/s8tmfisc/modelVal.pt')
        test_writer.add_scalar('mean EPE', val_loss / val_size, training_iterations)
    if stop:
        break
