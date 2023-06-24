#############
## imports ##
#############

import torch
import torch.nn as nn
from training.metrics import calculate_iou

#############
## globals ##
#############

PROB_IDX = 20
X_IDX = 21
Y_IDX = 22
W_IDX = 23
H_IDX = 24

#############
## classes ##
#############

class YOLOV1Loss(nn.Module):
    def __init__(self, split_size: int, num_classes: int, num_boxes: int, lambda_coord: float, lambda_noobj: float):
        """
        Computes the loss from the yolov1 paper
        Inputs:
            >> split_size: (int) Size of each grid in the splitted image (split_size, split_size)
            >> num_classes: (int) Number of classes in the dataset
            >> num_boxes: (int) Number of boxes predicted per each grid of the model output.
            >> lambda_coord: (float) Check loss function formula from the paper
            >> lambda_noobj: (float) Check loss function formula from the paper
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum") # In the paper they do not average it.
        self.S = split_size
        self.C = num_classes
        self.B = num_boxes
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        return

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Inputs:
            >> predictions: (torch.Tensor [B, S, S, classes * (5*model_out_boxes)]) TODO
            >> labels: (torch.Tensor [B, S, S, classes * 5])
        Outputs:
            >> loss: (torch tensor) Values of the yolov1 loss following the formula that apppears on the paper.
        """
        # The label shape for one image is: (S,S,25), where 20 are for the class predictions (if having 20 classes), 21 will be for the prob and the 4 remaining are for the bbox (x,y,w,h).
        # The predictions shape for one image is: (S,S,30), where 20 are for the class predictions (if having 20 classes), 21 will be for the prob and the 4 remaining are for the bbox. 25 will be for prob2 and the other four are for the second bbox.
        predictions = predictions.view(-1, self.S, self.S, self.C + self.B*5)

        # Get box responsible for each cell (bestbox)
        iou_b1 = calculate_iou(predictions[...,21:25], labels[...,21:25], box_format='midpoint').unsqueeze(dim=0)
        iou_b2 = calculate_iou(predictions[...,26:30], labels[...,21:25], box_format='midpoint').unsqueeze(dim=0)
        ious_results = torch.cat(tensors=(iou_b1, iou_b2), dim=0)
        _, bestbox = torch.max(ious_results, dim=0)
        thereis_object = labels[...,PROB_IDX].unsqueeze(dim=3) # (B,S,S,1) containing 0 if there is object, 1 if there is not.

        # Label: [c1,...,c20,pc,x,y,w,h]
        # Box loss (First two rows of the loss function in the paper)
        ## Get the box predictions where there is an object for the best box and labels
        box_predictions = thereis_object * ( (1-bestbox)*predictions[...,21:25] + bestbox*predictions[...,26:30])
        box_labels = thereis_object * labels[...,21:25]
        ## Calculate the square root for the w and h (1e-6 added for numerical stability). We want to keep the sign -> * torch.sign
        box_predictions[...,2:4] = torch.sqrt((box_predictions[...,2:4] + 1e-6).abs()) * torch.sign(box_predictions[...,2:4]) # +1e-6 for numerical stability
        box_labels[...,2:4] = torch.sqrt(box_labels[...,2:4])
        box_loss = self.mse(box_predictions, box_labels)

        # Objects loss (3rd row in the paper)
        ## Get the objects predictions where there is an object for the best box and labels
        object_predictions = thereis_object * ( (1-bestbox)*predictions[...,20:21] + bestbox*predictions[...,25:26])
        object_labels = thereis_object * labels[...,20:21]
        object_loss = self.mse(object_predictions, object_labels)

        # Not objects loss (4th row in the paper)
        ## Get the objects predictions where there is an object for the best box and labels
        noobject_predictions = (1 - thereis_object) * ( torch.cat(tensors=(predictions[...,20:21], predictions[...,25:26]), dim=-1))
        noobject_labels = (1 -thereis_object) * labels[...,20:21].repeat(1,1,1,2)
        noobject_loss = self.mse(noobject_predictions, noobject_labels)

        ## Probability loss (h,w)
        prob_predictions = thereis_object * predictions[...,0:20]
        prob_labels = thereis_object * labels[...,0:20]
        prob_loss = self.mse(prob_predictions, prob_labels)

        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * noobject_loss + prob_loss
        return loss
