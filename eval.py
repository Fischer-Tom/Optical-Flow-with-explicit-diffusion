import torch
import torch.nn as nn
from datasets.flying_chairs import FlyingChairsDataset
from torchvision import transforms
from models.FlowNetC import FlowNetC
from image_lib.core import display_flow_tensor


use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')


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

im1, im2, flow = next(iter(train_loader))


flowNet.load_state_dict(torch.load('model.pt', map_location='cpu'))
flowNet.eval()


pred_flow = flowNet(im1, im2)


display_flow_tensor(pred_flow[0].detach())
display_flow_tensor(flow[0])

display_flow_tensor(pred_flow[1].detach())
display_flow_tensor(flow[1])