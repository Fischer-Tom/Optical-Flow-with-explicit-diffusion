#!/usr/bin/env python3
import os
from os.path import join
from torch.utils.data import Dataset
from image_lib.io import read

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

class FlyingChairsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.IDs = [remove_suffix(file,'_flow.flo') for file in os.listdir(self.img_dir) if file.endswith('.flo')]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        path = join(self.img_dir, self.IDs[idx])
        im1 = read(f'{path}_img1.ppm')
        im2 = read(f'{path}_img2.ppm')
        flow = read(f'{path}_flow.flo')
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            flow = self.transform(flow)
        return im1, im2, flow
