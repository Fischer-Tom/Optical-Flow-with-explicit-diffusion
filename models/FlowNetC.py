import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowNetC(nn.Module):

    def __init__(self):
        super().__init__()
        self.ExtractorA = FeatureExtractor()
        self.ExtractorB = FeatureExtractor()
        self.conv_redir = SimpleConv(256, 32, 1, 1, 0)
        self.Corr = CorrelationModule()
        self.Encoder = Encoder()
        self.Decoder = Decoder()


    def forward(self, im1, im2):
        x1, corrA = self.ExtractorA(im1)
        _, corrB = self.ExtractorB(im2)
        corr = self.Corr(corrA, corrB)
        conv_redir = self.conv_redir(corrA)
        x2, x3, x4, x5 = self.Encoder(torch.cat((corr, conv_redir), dim=1))
        pred = self.Decoder([x1, x2, x3, x4, x5])
        return pred


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        dim = 64
        self.conv1 = SimpleConv(3, dim, 7, 2, 3, bias=False)
        self.conv2 = SimpleConv(dim, dim * 2, 5, 2, 2, bias=False)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 5, 2, 2, bias=False)

    def forward(self, x):
        x = self.conv2(self.conv1(x))

        return x, self.conv3(x)


class CorrelationModule(nn.Module):

    def __init__(self, k=0, d=20, s1=1, s2=2):
        super().__init__()
        self.k = k
        self.d = d
        self.s1 = s1
        self.s2 = s2

    def forward(self, x1, x2):
        pad = self.d // 2
        padded_x1 = F.pad(x1, (pad, pad, pad, pad))
        patches = padded_x1.unfold(2, self.d + self.s1, self.s1).unfold(3, self.d + self.s1, self.s1)
        b, d, w, h = x1.shape
        out = torch.zeros((b, (self.d + self.s1) ** 2, w, h))
        for i in range(w):
            for j in range(h):
                for k in range(b):
                    patch = patches[k, :, i, j]
                    weight = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(x2[k, :, i, j], -1), -1), 0)
                    pixel_corr = F.conv2d(patch, weight)
                    out[k, :, i, j] = pixel_corr.flatten()
        return out

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        dim = 64
        self.conv3_1 = SimpleConv(473, dim*4, 3, 1, 1, bias=False)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1, bias=False)
        self.conv4_1 = SimpleConv(dim * 8, dim * 8, 3, 1, 1, bias=False)
        self.conv5 = SimpleConv(dim * 8, dim * 8, 3, 2, 1, bias=False)
        self.conv5_1 = SimpleConv(dim * 8, dim * 8, 3, 1, 1, bias=False)
        self.conv6 = SimpleConv(dim * 8, dim * 16, 3, 2, 1, bias=False)

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
        self.deconv4 = SimpleUpConv(in_ch, 256, 1, 2, 0, 1, False)
        self.deconv3 = SimpleUpConv(256 + 512 + 2, 128, 1, 2, 0, 1, False)
        self.deconv2 = SimpleUpConv(128 + 256 + 2, 64, 1, 2, 0, 1, False)

        self.flow5 = nn.Conv2d(in_ch, 2, 5, 1, 2, bias=False)
        self.flow4 = nn.Conv2d(256 + 512 + 2, 2, 5, 1, 2, bias=False)
        self.flow3 = nn.Conv2d(128 + 256 + 2, 2, 5, 1, 2, bias=False)
        self.prediction = nn.Conv2d(64 + 128 + 2, 2, 5, 1, 2, bias=False)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        [x1, x2, x3, x4, x] = x

        x = torch.cat((self.deconv5(x), x4), dim=1)
        flow5 = self.flow5(x)
        x = torch.cat((self.deconv4(x), x3, self.upsample2(flow5)), dim=1)
        flow4 = self.flow4(x)
        x = torch.cat((self.deconv3(x), x2, self.upsample2(flow4)), dim=1)
        flow3 = self.flow3(x)
        x = torch.cat((self.deconv2(x), x1, self.upsample2(flow3)), dim=1)
        x = self.prediction(x)
        return self.upsample4(x)


class SimpleConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, pad_mode='reflect', bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                              stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class SimpleUpConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, output_padding=1, bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                                       stride=stride, padding=pad, output_padding=output_padding, bias=bias)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x
