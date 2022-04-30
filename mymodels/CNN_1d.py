#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=3):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=3),  # 16, 26 ,26
            nn.LayerNorm(510),
            # nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.LayerNorm(253),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            # nn.InstanceNorm1d(64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),  # 128,8,8
            # nn.InstanceNorm1d(64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(32),    # 64pu  32
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*32, 256),   # (64*64, 256)cwru   (64*64,1024)pu
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),     # (256,64)           (1024,1024)pu
            nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, out_channel),
    )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        # x = self.fc(x)

        return x
