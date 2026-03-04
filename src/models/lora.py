import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LoRAConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 r=16,
                 lora_alpha=16,
                 lora_dropout=0.3,
                 use_bias=True,
                 activation=True):
        super().__init__()

        # 1) frozen base Conv2d
        self.base_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias
        )
        self.base_conv.requires_grad_(False)

        # 2) LoRA branch: A(3x3) + B(1x1)
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(lora_dropout)

        if r > 0:
            self.lora_down = nn.Conv2d(
                in_channels,
                r,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            self.lora_up = nn.Conv2d(
                r,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            # LoRA B는 0으로 초기화
            nn.init.zeros_(self.lora_up.weight)
        else:
            self.lora_down = None
            self.lora_up = None

        self.use_activation = activation
        if activation:
            self.act = nn.ReLU()

    def forward(self, x):
        base_out = self.base_conv(x)

        if self.r > 0:
            lora = self.lora_down(x)
            lora = self.lora_up(lora)
            lora = self.dropout(lora) * self.scaling
            out = base_out + lora
        else:
            out = base_out

        if self.use_activation:
            out = self.act(out)
        return out


class LoRACNN(nn.Module):
    def __init__(self, num_classes=10, r=8, lora_alpha=16, lora_dropout=0.4):
        super().__init__()

        # conv1: baseline conv + relu + pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # conv2: LoRAConv2d + pool
        self.conv2_lora = LoRAConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            activation=True
        )
        self.pool2 = nn.MaxPool2d(2)

        # conv3: LoRAConv2d + pool
        self.conv3_lora = LoRAConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            activation=True
        )
        self.pool3 = nn.MaxPool2d(2)

        # GAP
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.conv2_lora(x)
        x = self.pool2(x)

        x = self.conv3_lora(x)
        x = self.pool3(x)

        x = self.gap(x)           # (B,128,1,1)
        x = x.view(x.size(0), -1) # (B,128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

