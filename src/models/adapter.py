import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bottleneck_ratio=2,
                 dropout=0.1,
                 activation=True,
                 use_bias=True):
        super().__init__()

        # 1) frozen base Conv2d (pretrained CNN의 conv 역할)
        self.base_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias
        )
        self.base_conv.requires_grad_(False)

        # 2) Adapter bottleneck: C -> r -> C (C = out_channels)
        self.out_channels = out_channels
        self.bottleneck_ratio = bottleneck_ratio
        r = max(out_channels // bottleneck_ratio, 1)

        self.down = nn.Conv2d(
            out_channels,  # base_out 채널
            r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.up = nn.Conv2d(
            r,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.dropout = nn.Dropout2d(dropout)

        self.use_activation = activation
        if activation:
            self.act = nn.ReLU()

    def forward(self, x):
        # base conv (frozen)
        base_out = self.base_conv(x)              # (B, C_out, H, W)

        # adapter path: base_out → down → ReLU → dropout → up
        h = self.down(base_out)                  # (B, r, H, W)
        h = F.relu(h)
        h = self.dropout(h)
        delta = self.up(h)                       # (B, C_out, H, W)

        out = base_out + delta                   # residual

        if self.use_activation:
            out = self.act(out)
        return out


# Adapter CNN class 정의
class AdapterCNN(nn.Module):
    def __init__(self, num_classes=10, bottleneck_ratio=2, dropout=0.3):
        super().__init__()

        # conv1: baseline conv + relu + pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # conv2: AdapterConv2d + pool
        self.conv2_adapter = AdapterConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bottleneck_ratio=bottleneck_ratio,
            dropout=0.1,
            activation=True
        )
        self.pool2 = nn.MaxPool2d(2)

        # conv3: AdapterConv2d + pool
        self.conv3_adapter = AdapterConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bottleneck_ratio=bottleneck_ratio,
            dropout=0.1,
            activation=True
        )
        self.pool3 = nn.MaxPool2d(2)

        # GAP
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.conv2_adapter(x)
        x = self.pool2(x)

        x = self.conv3_adapter(x)
        x = self.pool3(x)

        x = self.gap(x)                # (B,128,1,1)
        x = x.view(x.size(0), -1)      # (B,128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

