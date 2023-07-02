import torch
import torch.nn as nn

class Darknet19Backbone(nn.Module):

    def __init__(self, in_chs: int):
        """
        Resnet50Backbone
        Inputs:
            >> in_chs: (int) Number of channels in the input images
        Attributes:
            >> device_param: (nn.Parameter) Parameter to access to the devices where the model is running more easily.
            >> backbone: (nn.Parameter) Backbone model.
        """
        super(Darknet19Backbone, self).__init__()
        self.device_param = nn.Parameter(torch.empty(0))

        # (batch, 3, 448, 448) to (batch, 64, 112, 112)
        conv_layer1 = nn.Sequential(
            ConvLayer(in_chs, out_chs=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # (batch, 64, 112, 112) to (batch, 192, 56, 56)
        conv_layer2 = nn.Sequential(
            ConvLayer(in_chs=64, out_chs=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # (batch, 192, 56, 56) to (batch, 512, 28, 28)
        conv_layer3 = nn.Sequential(
            ConvBlockKer1x1Ker3x3(in_chs=192, l1_out_chs=128, l2_out_chs=256, total_blocks=1),
            ConvBlockKer1x1Ker3x3(in_chs=256, l1_out_chs=256, l2_out_chs=512, total_blocks=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # (batch, 512, 28, 28) to (batch, 1024, 14, 14)
        conv_layer4 = nn.Sequential(
            ConvBlockKer1x1Ker3x3(in_chs=512, l1_out_chs=256, l2_out_chs=512,  total_blocks=4),
            ConvBlockKer1x1Ker3x3(in_chs=512, l1_out_chs=512, l2_out_chs=1024, total_blocks=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # (batch, 1024, 14, 14) to (batch, 1024, 7, 7)
        conv_layer5 = nn.Sequential(
            ConvBlockKer1x1Ker3x3(in_chs=1024, l1_out_chs=512, l2_out_chs=1024,  total_blocks=2),
            ConvLayer(in_chs=1024, out_chs=1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_chs=1024, out_chs=1024, kernel_size=3, stride=2, padding=1),)

        # (batch, 1024, 7, 7) to (batch, 1024, 7, 7)
        conv_layer6 = nn.Sequential(
            ConvLayer(in_chs=1024, out_chs=1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_chs=1024, out_chs=1024, kernel_size=3, stride=1, padding=1),)

        self.backbone = nn.Sequential(
            conv_layer1,
            conv_layer2,
            conv_layer3,
            conv_layer4,
            conv_layer5,
            conv_layer6,)

    def forward(self, images: torch.tensor):
        """
        TODO
        Inputs:
            >> images: (torch.tensor [Batch, CHS, IMG_H, IMG_W])
        Outputs:
            >> features: (torc.tensor [Batch, 1024, 7, 7])
        """
        return self.backbone(images)

class ConvBlockKer1x1Ker3x3(nn.Module):

    def __init__(self, in_chs: int, l1_out_chs: int, l2_out_chs: int, total_blocks: int):
        """
        TODO
        Inputs:
            >> in_chs:       (int) Number of input channels
            >> l1_out_chs:   (int) Number of outputs channels after the first ConvLayer
            >> l2_out_chs:   (int) Number of outputs channels after the second ConvLayer
            >> total_blocks: (int) Number of times that the 1x1Conv + 3x3Conv block will be repeated
        Attributes:
            >>
        """
        super(ConvBlockKer1x1Ker3x3, self).__init__()

        self.conv_block = nn.Sequential()
        for i in range(total_blocks):
            self.conv_block.add_module(
                f'{i}_1x1_and_3x3_conv_block',
                nn.Sequential(
                    ConvLayer(in_chs,     l1_out_chs, kernel_size=1, stride=1, padding=0),
                    ConvLayer(l1_out_chs, l2_out_chs, kernel_size=1, stride=1, padding=0),))
            in_chs = l2_out_chs

    def forward(self, inputs: torch.tensor):
        """
        Performs the forward step for the convolutional layer
        Inputs:
            >> inputs: (torch.tensor [batch, in_chs, h, w])
        Outputs:
            >> outputs: (torch.tensor [batch, l2_out_chs, floor(h-2), floor(w-2)])
        """
        return self.conv_block(inputs)

class ConvLayer(nn.Module):

    def __init__(self, in_chs: int, out_chs: int, kernel_size: int, stride: int, padding: int):
        """
        A convolutional layer followed by BatchNorm2d and LeakyReLU.
        Inputs:
            >> in_chs: (int) Number of input chs
            >> out_chs: (int) Number of out chs
            >> kernel_size: (int) Kernel size for the convolution
            >> stride: (int) Stride to be applied during the convolution
            >> padding: (int) Padding to be applied during the convolution
        Attributes:
            >> layer: (torch.Module) Convolution layer + Batch norm + LeakyReLU
        """
        super(ConvLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=False), # Do not add bias as the next layer is a batch norm layer
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(0.2))

    def forward(self, inputs: torch.tensor):
        """
        Performs the forward step for the convolutional layer
        Inputs:
            >> inputs: (torch.tensor [batch, in_chs, h, w])
        Outputs:
            >> outputs: (torch.tensor [batch, in_chs, floor((h + 2*padding - kernel_size) / stride) + 1, floor((w + 2*padding - kernel_size) / stride) + 1])
        """
        return self.layer(inputs)

