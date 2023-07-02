import torch
import torch.nn as nn
import models.backbones as backbones

"""
YOLO algorithm: (All the following applies for every cell)
    The image is splitted into a grid of SxS cells. Each cell will output a predictions with the corresponding bounding box.
    We just one one bounding box for each object in the image, so do we make sure that only one bbox will be outputed for each object.
    The idea is that we will find one cell responsible for outputing that object. That responsible cell is the one that contains the 
    objects midpoints.

    Each of these cells will start with (0,0) at the top-left corner and bottom-right will be (1,1). EACH OUTPUT AN LABEL WILL BE
    RELATIVE TO THE CELL. Each bbox for each cell will have:
        [x,y,w,h], where (x,y) is the coordinates of the midpoint and w,h are the width and height of the bbox.
        x,y will be [0,1]
        w,h can be larger than 1 if the object is wider and/or taller than the cell.

Shape of the labels:
    label_cell = [c1,c2,...,c20,p_c,x,y,w,h]
    where cx refers to the different classes,
    p_c to the probability that there is an object in that cell (1 or 0).

Shape of the predictions:
    Predictions will look very similar, but we will output two bboxes, so that they will specialize to output different bounding boxes (wide vs tall).
    pred_cell = [c1,c2,c..,c20,p_c1,x1,y1,w1,h1,p_c2,x2,y2,w2,h2]
    where:
        cx refers to the class.
        p_c1 refers to the probability that tere is an object for bbox 1.
        x1,y1,w1,h1 is the bbox 1
        p_c2 refers to the probability that tere is an object for bbox 2.
        x2,y2,w2,h2 is the bbox 2

Note: A CELL CAN ONLY DECTECT ONE OBJECT. This is a limitation of yolo.

The target shape for one image is: (S,S,25), where 20 are for the class predictions (if having 20 classes), 21 will be for the prob and the 4 remaining are for the bbox.
The predictions shape for one image is: (S,S,30), where 20 are for the class predictions (if having 20 classes), 21 will be for the prob and the 4 remaining are for the bbox. 25 will be for prob2 and the other four are for the second bbox.
"""

DATA_PER_BOX = 5 # The 5 values are: (prob,x,y,w,h)
AVAILABLE_BACKBONES = ['resnet50', 'resnet34', 'resnet18', 'darknet19']

class YOLOV1(nn.Module):

    def __init__(self, in_chs: int, num_classes: int, split_size: int, num_boxes: int, backbone_to_use: str):
        """
        YOLOV1 pytorch implementation. It is possible to use different backbones as the original darknet19 or different
        models of resnet (resnet18, resnet34, resnet50).
        Inputs:
            >> in_chs: (int) Number of channels in the input images
            >> num_classes: (int) Number of classes present in the dataset.
            >> split_size: (int) Size of each cell when splitting the image.
            >> num_boxes: (int) Number of boxes per cell.
        Attributes:
            >> model: (nn.Module) YOLOV1 model.
        """
        super(YOLOV1, self).__init__()
        self.device_param = nn.Parameter(torch.empty(0))

        backbone, backbone_out_feat = load_backbone(backbone_to_use, in_chs)
        fcl = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_feat * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * DATA_PER_BOX))) # (S,S,30) where (num_classes + num_boxes * 5) = 30, and 5 is for (prob,x,y,w,h)

        self.model = nn.Sequential(
            backbone,
            fcl)

    def forward(self, images: torch.tensor):
        """
        Performs the forward step for yolov1.
        Inputs:
            >> images: (torch.tensor [Batch, CHS, IMG_H, IMG_W])
        Outputs:
            >> predictions: (torch.tensor [Batch, S*S*(num_classes + num_boxes * DATA_PER_BOX)])
        """
        return self.model(images)

def load_backbone(backbone_name: str, in_chs: int):
    """
    Loads the especified backbone.
    Inputs:
        >> backbone_name: (str) Name of the backbone (resnet50, resnet34, resnet18 or darknet19)
        >> in_chs: (int) Quantity of input chs.
    Outputs:
        >> backbone: (nn.Module) Backbone network
        >> backbone_out_feat: (int) Number of output chs by the backbone
    """
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
        raise ValueError(f'{backbone_name} is not a valid option. You can choose between {AVAILABLE_BACKBONES}.')

    return backbone, backbone_out_feat
