import torch
import flow_vis
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from PIL import Image
from image_lib.io import read
from torchvision import transforms, utils


def image_to_tensor(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
        print(im_np.shape)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return torch.from_numpy(im_np).permute(2, 0, 1)


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
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=2,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad = False
        self.init_weights()

    def forward(self, x, time_steps=20):
        for _ in range(time_steps):
            x = self.conv(x)
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


tens1 = torch.Tensor(read('../datasets/FlyingChairs_release/data/00001_flow.flo')).permute(2, 0, 1)
tens2 = torch.Tensor(read('../datasets/FlyingChairs_release/data/00002_flow.flo')).permute(2, 0, 1)
tens = torch.stack((tens1, tens2), dim=0)
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
tens = image_to_tensor('face.jpg', mode='GS').float()

mask = torch.FloatTensor(tens.shape).uniform_() > 0.9
# tens = hom_diff_inpaint_gs(tens*mask, mask, 20)
block = InpaintingBlock()
diffused_tens = block(tens*mask, 100)
diffused_tens = diffused_tens.detach()
display_tensor(diffused_tens)
"""
