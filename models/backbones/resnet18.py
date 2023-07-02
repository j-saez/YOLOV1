import torch
import torchvision
import torch.nn as nn

class Resnet18Backbone(nn.Module):

    def __init__(self, in_chs: int):
        """
        Resnet18Backbone
        Inputs:
            >> in_chs: (int) Number of channels in the input images
        Attributes:
            >> device_param: (nn.Parameter) Parameter to access to the devices where the model is running more easily.
            >> backbone: (nn.Parameter) Backbone model.
        """
        super(Resnet18Backbone, self).__init__()
        self.device_param = nn.Parameter(torch.empty(0))

        resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential()
        self.backbone.add_module('restnet', nn.Sequential(*list(resnet18.children()))[:-2])
        self.backbone.add_module('last_layer', nn.MaxPool2d(kernel_size=2, stride=2)) 
        return

    def forward(self, images: torch.tensor):
        """
        TODO
        Inputs:
            >> images: (torch.tensor [Batch, CHS, IMG_H, IMG_W])
        Outputs:
            >> features: (torc.tensor [Batch, 2048, 7, 7])
        """
        return self.backbone(images)

if __name__ == "__main__":
    inputs = torch.rand(4,3,448,448)
    model = Resnet18Backbone(in_chs=3)
    outputs = model(inputs)
    print(f'resnet18 outputs shape = {outputs.shape}')
