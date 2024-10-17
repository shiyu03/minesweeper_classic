import torch
from torch import nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv_start = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_middle2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_middle3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_middle4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_middle5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_middle6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_middle7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_middle8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv_middle9 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv_middle10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_middle11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_middle12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_middle13 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_middle14 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_middle15 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_end = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv_start(x))
        x = torch.relu(self.conv_middle2(x))
        x = torch.relu(self.conv_middle3(x))
        x = torch.relu(self.conv_middle4(x))
        x = torch.relu(self.conv_middle5(x))
        x = torch.relu(self.conv_middle6(x))
        x = torch.relu(self.conv_middle7(x))
        x = torch.relu(self.conv_middle8(x))
        # x = torch.relu(self.conv_middle9(x))
        # x = torch.relu(self.conv_middle10(x))
        # x = torch.relu(self.conv_middle11(x))
        # x = torch.relu(self.conv_middle12(x))
        # x = torch.relu(self.conv_middle13(x))
        # x = torch.relu(self.conv_middle14(x))
        # x = torch.relu(self.conv_middle15(x))
        x = self.conv_end(x)
        x = self.flatten(x)
        return x
