import torch
import torch.nn as nn



class FlowNetS(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, im1, im2):
        stacked_images = torch.cat((im1, im2), 1)
        encoder_out = self.encoder(stacked_images)
        out = self.decoder(encoder_out)
        return out


class Encoder(nn.Module):

    def __init__(self, in_ch=6):
        super().__init__()
        dim = 64
        self.conv1 = SimpleConv(in_ch, dim, 7, 2, 3, bias=False)
        self.conv2 = SimpleConv(dim, dim * 2, 5, 2, 2, bias=False)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 5, 2, 2, bias=False)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1, bias=False)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1, bias=False)
        self.conv4_1 = SimpleConv(dim * 8, dim * 8, 3, 1, 1, bias=False)
        self.conv5 = SimpleConv(dim * 8, dim * 8, 3, 2, 1, bias=False)
        self.conv5_1 = SimpleConv(dim * 8, dim * 8, 3, 1, 1, bias=False)
        self.conv6 = SimpleConv(dim * 8, dim * 16, 3, 2, 1, bias=False)

    def forward(self, x):
        x1 = self.conv2(self.conv1(x))
        x2 = self.conv3_1(self.conv3(x1))
        x3 = self.conv4_1(self.conv4(x2))
        x4 = self.conv5_1(self.conv5(x3))
        x5 = self.conv6(x4)

        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):

    def __init__(self, in_ch=1024):
        super().__init__()
        self.deconv5 = SimpleUpConv(in_ch, 512, 1, 2, 0, 1, False)
        self.deconv4 = SimpleUpConv(in_ch, 256, 1, 2, 0, 1, False)
        self.deconv3 = SimpleUpConv(256+512+2, 128, 1, 2, 0, 1, False)
        self.deconv2 = SimpleUpConv(128+256+2, 64, 1, 2, 0, 1, False)

        self.flow5 = nn.Conv2d(in_ch, 2, 5, 1, 2, bias=False)
        self.flow4 = nn.Conv2d(256+512+2, 2, 5, 1, 2, bias=False)
        self.flow3 = nn.Conv2d(128+256+2, 2, 5, 1, 2, bias=False)
        self.prediction = nn.Conv2d(64+128+2, 2, 5, 1, 2, bias=False)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.sig = nn.Sigmoid()

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

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, output_padding=1,bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                                       stride=stride, padding=pad, output_padding=output_padding, bias=bias)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

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
