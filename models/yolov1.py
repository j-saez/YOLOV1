import torch
import torch.nn as nn
from models.utils import load_conv_layers_from_configuration

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

#############
## classes ##
#############

class YOLOV1(nn.Module):

    # In the original paper: num_classes = 20 (COCO dataset), split_size = 7, num_boxes = 2
    def __init__(self, in_chs: int, num_classes: int, split_size: int, num_boxes: int, conf_file: str='./models/configurations/default.txt', **kwargs):
        """
        YOLO v1 implementation
        Inputs:
            >> in_chs: (int) number of input channels
            >> num_classes: (int) number of classes
            >> split_size: (int) Size for each reagion when splitting the image. 
            >> num_boxes: (int) Quantity of boxes per grid
            >> conf_file: (str)

        Attributes:
            >> device_param: (nn.Parameter) Parameter to access to the devices where the model is running more easily.
            >> convs: (nn.ModuleList) Convolutional layers for YOLO v1.
            >> fcl: (nn.Sequential) Fully connected layer for YOLO v1.
        """
        super(YOLOV1, self).__init__()
        self.device_param = nn.Parameter(torch.empty(0))

        self.convs = load_conv_layers_from_configuration(in_chs, conf_file)
        self.fcl = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024 * split_size * split_size, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * DATA_PER_BOX))) # (S,S,30) where (num_classes + num_boxes * 5) = 30, and 5 is for (prob,x,y,w,h)
        return

    def forward(self, x):
        """
        Inputs:
            >> x: (torch.Tensor [B, CHS, H, W])
        Outpus:
            >> x: (torch.Tensor [B, split_size * split_size * (num_classes + num_boxes * 5)])
        """
        x = self.convs(x)
        x = self.fcl(x)
        return x
