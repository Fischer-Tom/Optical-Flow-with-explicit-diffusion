import torch
import flow_vis
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from image_lib.io import read
from torchvision import transforms, utils


def image_to_tensor(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return torch.from_numpy(im_np).permute(2, 0, 1)

def image_to_numpy(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return im_np


def tensor_to_image(tensor):
    pass


def display_image(path):
    pass


def display_flow_tensor(tensor):
    if len(tensor.shape) == 4:
        display_batch_flow_tensor(tensor)
    else:
        np_flow = tensor.permute((1, 2, 0)).numpy()
        flow_color = flow_vis.flow_to_color(np_flow, convert_to_bgr=False)
        plt.imshow(flow_color)
        plt.axis('off')
        plt.show()

def display_flow_numpy(np_array):
    flow_color = flow_vis.flow_to_color(np_array, convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.axis('off')
    plt.show()

def display_batch_flow_tensor(tensor):
    raise Exception("not implemented!")


def display_tensor(tensor):
    if tensor.shape[0] == 3:
        plt.imshow(tensor.permute(1, 2, 0))
    else:
        plt.imshow(tensor.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.show()


def hom_diff_gs(image, diff_steps, tau=0.25, h1=1, h2=1):
    assert (tau <= 0.25)
    width = image.shape[1]
    height = image.shape[2]
    hx = tau / (h1 * h1)
    hy = tau / (h2 * h2)
    for t in range(diff_steps):
        copy = image
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                image[0][i][j] = (1 - 2 * hx - 2 * hy) * copy[0][i][j] + hx * copy[0][i - 1][j] + \
                                 hx * copy[0][i + 1][j] + hy * copy[0][i][j - 1] + hy * copy[0][i][j + 1]
    return image


def hom_diff_inpaint_gs(image, mask, diff_steps, tau=0.25, h1=1, h2=1):
    assert mask.shape == image.shape

    width = image.shape[1]
    height = image.shape[2]
    hx = tau / (h1 * h1)
    hy = tau / (h2 * h2)
    for t in range(diff_steps):
        copy = image
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if mask[0][i][j]:
                    image[0][i][j] = copy[0][i][j]
                else:
                    image[0][i][j] = (1 - 2 * hx - 2 * hy) * copy[0][i][j] + hx * copy[0][i - 1][j] + \
                                     hx * copy[0][i + 1][j] + hy * copy[0][i][j - 1] + hy * copy[0][i][j + 1]
    return image


class DiffusionBlock(nn.Module):

    def __init__(self):
        super(DiffusionBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=0, groups=1, bias=False)
        self.conv2 = nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=3,padding=0, bias=False)
        self.conv.requires_grad = False
        self.conv2.requires_grad = False
        self.init_weights()

    def forward(self, x, time_steps=1):
        b,c,h,w = x.shape
        z = x.flatten(start_dim = 2)
        for _ in range(time_steps):
            y = z
            y = self.conv2(self.conv(y))
            z = y

        return z.reshape((b,c,h,w))

    def init_weights(self, tau=0.25, h1=1, h2=1):
        hx = tau / (h1 * h1)
        hy = tau / (h2 * h2)
        hx = 1
        hy=1
        weight = torch.zeros_like(self.conv.weight)
        weight[0,0,0] = 1
        weight[0,0,1] = -2
        weight[0,0,2] = 1
        weight[0,1,0] = 1
        weight[0,1,1] = -2
        weight[0,1,2] = 1
        self.conv.weight = nn.Parameter(weight)
        self.conv2.weight = nn.Parameter(weight)

class InpaintingBlock(nn.Module):

    def __init__(self):
        super(InpaintingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=2,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad = False
        self.init_weights()

    def forward(self, x, time_steps=20):
        masked_x = x
        mask = x == 0
        for _ in range(time_steps):
            diff_x = self.conv(x)
            x = masked_x + diff_x*mask
        return x

    def init_weights(self, tau=0.25, h1=1, h2=1):
        hx = tau / (h1 * h1)
        hy = tau / (h2 * h2)
        weight = torch.zeros_like(self.conv.weight)
        weight[0][0][1][0] = hx
        weight[0][0][1][2] = hx
        weight[0][0][0][1] = hy
        weight[0][0][2][1] = hy
        weight[0][0][1][1] = (1 - 2 * hx - 2 * hy)
        weight[1][0][1][0] = hx
        weight[1][0][1][2] = hx
        weight[1][0][0][1] = hy
        weight[1][0][2][1] = hy
        weight[1][0][1][1] = (1 - 2 * hx - 2 * hy)
        self.conv.weight = nn.Parameter(weight)

class GausConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.init_weights(15,1)



    def forward(self,x):
        kernel = torch.outer(self.weight, self.weight).repeat(2, 1, 1, 1)
        return F.conv2d(x, kernel, padding=7, groups=2)


    def init_weights(self, kernel_size, sigma):
        n = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        sig2 = 2 * sigma * sigma
        w = torch.exp(-n ** 2 / sig2)
        self.weight = nn.Parameter(w / torch.sum(w), requires_grad=True)


tens1 = torch.Tensor(read('../datasets/FlyingChairs_release/data/00001_flow.flo')).permute(2, 0, 1)
tens2 = torch.Tensor(read('../datasets/FlyingChairs_release/data/00002_flow.flo')).permute(2, 0, 1)
noise_tens = tens1 + (200**0.5)*torch.randn(tens2.shape)
tens = torch.stack((tens1, tens2), dim=0)
device = torch.device('cuda')
Gaus = GausConvLayer().to(device)
noise_tens = noise_tens.to(device)
tens1 = tens1.to(device)
display_flow_tensor(noise_tens.cpu().detach())
optim = torch.optim.Adam(Gaus.parameters(), lr=1e-2, betas=(0.9,0.999), weight_decay=4e-4)
for i in range(1000):
    optim.zero_grad()
    x = Gaus(noise_tens)
    loss = torch.norm(x - tens1, p=2, dim=1).mean()
    loss.backward()
    optim.step()
print(Gaus.weight)
display_flow_tensor(x.cpu().detach())
"""
display_flow_tensor(tens[0])
block = DiffusionBlock()
diff_tens = block(tens)
display_flow_tensor(diff_tens[0].detach())
masc = torch.FloatTensor(1, 1, tens.shape[2], tens.shape[3]).uniform_() > 0.9
masc = masc.repeat(1, 2, 1, 1)
block = InpaintingBlock()
inp_tens = block(tens * masc, 50)
masked_tens = inp_tens * masc
display_flow_tensor(masked_tens[0].detach())
display_flow_tensor(inp_tens[0].detach())
"""