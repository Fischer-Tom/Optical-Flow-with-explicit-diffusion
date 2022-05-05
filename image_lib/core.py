import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from PIL import Image
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
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad=False
        self.init_weights()
        print(self.conv.weight)

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
        self.conv.weight = nn.Parameter(weight)

class InpaintingBlock(nn.Module):

    def __init__(self):
        super(InpaintingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad=False
        self.init_weights()

    def forward(self, x, time_steps=20):
        mask = x > 0
        masked_x = torch.masked_select(x.detach(),mask)
        for _ in range(time_steps):
            x = self.conv(x)
            x.masked_scatter_(mask, masked_x)
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
        self.conv.weight = nn.Parameter(weight)


tens = image_to_tensor('face.jpg', mode='GS').float()

mask = torch.FloatTensor(tens.shape).uniform_() > 0.9
# tens = hom_diff_inpaint_gs(tens*mask, mask, 20)
block = InpaintingBlock()
diffused_tens = block(tens*mask, 100)
diffused_tens = diffused_tens.detach()
display_tensor(diffused_tens)
