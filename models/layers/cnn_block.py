import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, **kwargs):
        """
        Convolutional block for yolo v1.
        Inputs:
            >> in_chs: (int) Number of input chs.
            >> out_chs: (int) Number of output chs.
        """
        super(CNNBlock, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, bias=False, **kwargs),
                nn.BatchNorm2d(out_chs),
                nn.LeakyReLU(0.1))

        return

    def forward(self, x):
        return self.model(x)
