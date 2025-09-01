import torch.nn as nn
from typing import Dict
from models import backbones

def load(conf: Dict):
    """
    Loads the especified backbone.
    Inputs:
        >> backbone_name: (str) Name of the backbone (resnet50, resnet34, resnet18 or darknet19)
        >> in_chs: (int) Quantity of input chs.
    Outputs:
        >> backbone: (nn.Module) Backbone network
        >> backbone_out_feat: (int) Number of output chs by the backbone
    """
    in_chs = conf["dataset"]["data_chs"]
    backbone_name = conf["model"]["backbone"]

    backbone = nn.Module()
    backbone_out_feat = -1


    if backbone_name == 'darknet19':
        backbone = backbones.Darknet19Backbone(in_chs)
        backbone_out_feat = 1024

    elif backbone_name == 'resnet50':
        backbone = backbones.Resnet50Backbone(in_chs)
        backbone_out_feat = 2048

    elif backbone_name == 'resnet34':
        backbone = backbones.Resnet34Backbone(in_chs)
        backbone_out_feat = 512

    elif backbone_name == 'resnet18':
        backbone = backbones.Resnet34Backbone(in_chs)
        backbone_out_feat = 512

    else:
        raise ValueError(f'{backbone_name} is not a valid option.')

    return backbone, backbone_out_feat
