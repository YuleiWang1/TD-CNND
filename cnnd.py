import torch
import math
from torch import nn
from numpy import *
from torch.nn import Conv1d, Linear, LeakyReLU, Dropout, Tanh, Sigmoid, ReLU


def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    )


class CNND(nn.Module):
    def __init__(self, bs):
        super(CNND, self).__init__()
        self.bs = bs
        self.conv1 = Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        # block1 ConvBlock(20, 30)
        self.conv2 = Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        # block2 ConvBlock(30, 40)
        self.conv5 = Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        # block3 ConvBlock(40, 30)
        self.conv8 = Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = Conv1d(512, 256, kernel_size=3, stride=2, padding=1)
        # block4 ConvBlock(30, 20)
        self.conv11 = Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv12 = Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv13 = Conv1d(256, 128, kernel_size=3, stride=2, padding=1)
        # block5 ConvBlock(20, 20)
        self.conv14 = Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv15 = Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv16 = Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        # I think the concept of avgpooling in original paper was wrong, so I replace it with 1x1 conv.
        self.pool = nn.Conv1d(64, 1, 1)
        num = int(ceil(ceil(ceil(ceil(ceil(self.bs/2)/2)/2)/2)/2))
        self.line = nn.Linear(num, 1)
        # 224 bands->7, 189 bands->6, 188bands=6
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # block1
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        # block2
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        # block3
        x = self.conv8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.relu(x)
        # # block4
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.relu(x)
        # block5
        x = self.conv14(x)
        x = self.relu(x)
        x = self.conv15(x)
        x = self.relu(x)
        x = self.conv16(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.line(x.flatten(1))
        x = self.sigmod(x)
        # print(x.shape)
        return x.flatten()



if __name__ == "__main__":
    input = torch.rand(1, 1, 189)
    model = CNND(bs=189)
    output = model(input)
    # print(input.shape)
    # print(output.shape)



