import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import spatial_correlation_sample
import time


class FlowNetC(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.Extractor = FeatureExtractor()
        self.conv_redir = SimpleConv(256, 32, 1, 1, 0)
        self.Encoder = Encoder(in_ch=473)
        self.Decoder = Decoder()

    def forward(self, im1, im2):
        x1, corrA = self.Extractor(im1)
        _, corrB = self.Extractor(im2)

        corr = spatial_correlation_sample(corrA,
                                          corrB,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation=1,
                                          dilation_patch=2)
        b, ph, pw, h, w = corr.size()
        corr = corr.view(b, ph*pw, h, w)/im1.size(1)
        corr = F.leaky_relu(corr, 0.1)
        conv_redir = self.conv_redir(corrA)
        x2, x3, x4, x5 = self.Encoder(torch.cat((corr, conv_redir), dim=1))
        pred = self.Decoder([x1, x2, x3, x4, x5])
        return pred


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = SimpleConv(3, 64, 7, 2, 3, bias=True)
        self.conv2 = SimpleConv(64, 128, 5, 2, 2, bias=True)
        self.conv3 = SimpleConv(128, 256, 5, 2, 2, bias=True)

    def forward(self, x):
        x = self.conv2(self.conv1(x))

        return x, self.conv3(x)


class CorrelationModule(nn.Module):

    def __init__(self, k=0, d=20, s1=1, s2=2, device='cpu'):
        super().__init__()
        self.k = k
        self.d = d
        self.s1 = s1
        self.s2 = s2
        self.device = device

    def forward(self, x1, x2):
        pad = self.d // 2
        padded_x2 = F.pad(x2, (pad, pad, pad, pad), mode='constant', value=0.0)

        patches = padded_x2.unfold(2, self.d + self.s1, self.s1).unfold(3, self.d + self.s1, self.s1).detach()
        b, d, w, h = x1.shape
        out = torch.zeros((b, (self.d + self.s1) ** 2, w, h)).to(self.device)
        for i in range(w):
            for j in range(h):
                for k in range(b):
                    patch = patches[k, :, i, j]
                    weight = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(x1[k, :, i, j], -1), -1), 0)
                    pixel_corr = F.conv2d(patch, weight)
                    out[k, :, i, j] = pixel_corr.flatten()
        return out


class CorrFast(nn.Module):
    def __init__(self, kernel_size=1, d=20, s1=1, s2=2):
        super().__init__()
        self.ks = kernel_size
        self.s1 = s1
        self.s2 = s2
        self.d = d
        self.padlayer = nn.ConstantPad2d(d, 0)

    def forward(self, feat1, feat2):
        feat2_pad = self.padlayer(feat2)
        b, _, height, width = feat1.shape
        offsetx, offsety = torch.meshgrid([torch.arange(0, height),
                                           torch.arange(0, width)], indexing='ij')
        output = torch.cat([
            torch.sum(
                feat1[:, :, dx:dx+1, dy:dy+1] *
                feat2_pad[:, :, dx:dx + 2 * self.d + 1:self.s2, dy:dy + 2 * self.d + 1:self.s2], 1,
                keepdim=True).flatten(start_dim=2)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
        ], 1)
        out_channels = output.shape[2]
        return output.reshape((b, out_channels, height, width))

class Correlation(nn.Module):
    def __init__(self, kernel_size=1, d=20, s1=1, s2=2):
        super().__init__()
        self.ks = kernel_size
        self.s1 = s1
        self.s2 = s2
        self.d = d
        self.padlayer = nn.ConstantPad2d(d, 0)

    def forward(self, feat1, feat2):
        feat2_pad = self.padlayer(feat2)
        _, _, height, width = feat1.shape
        offsetx, offsety = torch.meshgrid([torch.arange(0, 2*self.d + 1, step=self.s2),
                                           torch.arange(0, 2*self.d + 1, step=self.s2)], indexing='ij')

        output = torch.cat([

            torch.sum(
                feat1 *
                feat2_pad[:, :, dx:dx + height, dy:dy + width], 1,
                keepdim=True)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))

        ], 1)
        return output


class Encoder(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.conv3_1 = SimpleConv(in_ch, 256, 3, 1, 1, bias=True)
        self.conv4 = SimpleConv(256, 512, 3, 2, 1, bias=True)
        self.conv4_1 = SimpleConv(512, 512, 3, 1, 1, bias=True)
        self.conv5 = SimpleConv(512, 512, 3, 2, 1, bias=True)
        self.conv5_1 = SimpleConv(512, 512, 3, 1, 1, bias=True)
        self.conv6 = SimpleConv(512, 1024, 3, 2, 1, bias=True)
        self.conv6_1 = SimpleConv(1024, 1024, 3, 2, 1, bias=True)

    def forward(self, x):
        x2 = self.conv3_1(x)
        x3 = self.conv4_1(self.conv4(x2))
        x4 = self.conv5_1(self.conv5(x3))
        x5 = self.conv6(x4)
        return [x2, x3, x4, x5]


class Decoder(nn.Module):

    def __init__(self, in_ch=1024):
        super().__init__()
        self.deconv5 = SimpleUpConv(in_ch, 512, 1, 2, 0, 1, False)
        self.deconv4 = SimpleUpConv(in_ch+2, 256, 1, 2, 0, 1, False)
        self.deconv3 = SimpleUpConv(256 + 512 + 2, 128, 1, 2, 0, 1, False)
        self.deconv2 = SimpleUpConv(128 + 256 + 2, 64, 1, 2, 0, 1, False)

        self.flow6 = nn.Conv2d(in_ch, 2, 3, 1, 1, bias=False)
        self.flow5 = nn.Conv2d(in_ch+2, 2, 3, 1, 1, bias=False)
        self.flow4 = nn.Conv2d(256 + 512 + 2, 2, 3, 1, 1, bias=False)
        self.flow3 = nn.Conv2d(128 + 256 + 2, 2, 3, 1, 1, bias=False)
        self.prediction = nn.Conv2d(64 + 128 + 2, 2, 3, 1, 1, bias=False)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        [x1, x2, x3, x4, x] = x

        flow6 = self.flow6(x)
        x = torch.cat((self.deconv5(x), x4, self.upsampled_flow6_to_5(flow6)), dim=1)
        flow5 = self.flow5(x)
        x = torch.cat((self.deconv4(x), x3, self.upsampled_flow5_to_4(flow5)), dim=1)
        flow4 = self.flow4(x)
        x = torch.cat((self.deconv3(x), x2, self.upsampled_flow4_to_3(flow4)), dim=1)
        flow3 = self.flow3(x)
        x = torch.cat((self.deconv2(x), x1, self.upsampled_flow3_to_2(flow3)), dim=1)
        x = self.prediction(x)

        if self.training:
            return [x, flow3, flow4, flow5, flow6]
        return x


class SimpleConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, pad_mode='zeros', bias=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                              stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x))


class SimpleUpConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, output_padding=1,
                 bias=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                                       stride=stride, padding=pad, output_padding=output_padding, bias=bias)
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x))
