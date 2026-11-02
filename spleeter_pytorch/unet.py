# PyTorch U-Net for Spleeter stem separation.
# Architecture ported from the original Deezer TF model by Fangjun Kuang (Xiaomi Corp, 2023).
# Weights are loaded from the original TF checkpoint via checkpoint_reader.py.

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = nn.Conv2d(2,   16,  kernel_size=5, stride=2, padding=0)
        self.bn    = nn.BatchNorm2d(16,  track_running_stats=True, eps=1e-3, momentum=0.01)
        self.conv1 = nn.Conv2d(16,  32,  kernel_size=5, stride=2, padding=0)
        self.bn1   = nn.BatchNorm2d(32,  track_running_stats=True, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(32,  64,  kernel_size=5, stride=2, padding=0)
        self.bn2   = nn.BatchNorm2d(64,  track_running_stats=True, eps=1e-3, momentum=0.01)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=5, stride=2, padding=0)
        self.bn3   = nn.BatchNorm2d(128, track_running_stats=True, eps=1e-3, momentum=0.01)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0)
        self.bn4   = nn.BatchNorm2d(256, track_running_stats=True, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=0)

        self.up1  = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.bn5  = nn.BatchNorm2d(256, track_running_stats=True, eps=1e-3, momentum=0.01)
        self.up2  = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2)
        self.bn6  = nn.BatchNorm2d(128, track_running_stats=True, eps=1e-3, momentum=0.01)
        self.up3  = nn.ConvTranspose2d(256, 64,  kernel_size=5, stride=2)
        self.bn7  = nn.BatchNorm2d(64,  track_running_stats=True, eps=1e-3, momentum=0.01)
        self.up4  = nn.ConvTranspose2d(128, 32,  kernel_size=5, stride=2)
        self.bn8  = nn.BatchNorm2d(32,  track_running_stats=True, eps=1e-3, momentum=0.01)
        self.up5  = nn.ConvTranspose2d(64,  16,  kernel_size=5, stride=2)
        self.bn9  = nn.BatchNorm2d(16,  track_running_stats=True, eps=1e-3, momentum=0.01)
        self.up6  = nn.ConvTranspose2d(32,  1,   kernel_size=5, stride=2)
        self.bn10 = nn.BatchNorm2d(1,   track_running_stats=True, eps=1e-3, momentum=0.01)
        self.up7  = nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding=3)

    def forward(self, x):
        in_x = x  # (T, 2, 512, 1024)
        def pad(t): return F.pad(t, (1, 2, 1, 2), "constant", 0)
        def crop(t): return t[:, :, 1:-2, 1:-2]

        act = F.elu  # 4-stems model trained with ELU activations

        # Encoder: conv → BN → ELU (same order as TF model)
        c1 = act(self.bn(self.conv(pad(x))))
        c2 = act(self.bn1(self.conv1(pad(c1))))
        c3 = act(self.bn2(self.conv2(pad(c2))))
        c4 = act(self.bn3(self.conv3(pad(c3))))
        c5 = act(self.bn4(self.conv4(pad(c4))))
        c6 = self.conv5(pad(c5))  # no BN/act for bottleneck (TF output is discarded)

        u1 = self.bn5(act(crop(self.up1(c6))))
        u2 = self.bn6(act(crop(self.up2(torch.cat([c5, u1], 1)))))
        u3 = self.bn7(act(crop(self.up3(torch.cat([c4, u2], 1)))))
        u4 = self.bn8(act(crop(self.up4(torch.cat([c3, u3], 1)))))
        u5 = self.bn9(act(crop(self.up5(torch.cat([c2, u4], 1)))))
        u6 = self.bn10(act(crop(self.up6(torch.cat([c1, u5], 1)))))
        return torch.sigmoid(self.up7(u6)) * in_x
