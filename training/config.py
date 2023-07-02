import torch
import argparse
from dataclasses import dataclass


@dataclass
class Generalparams:
    pretrained_weights: str
    test_model_epoch: int
    workers: int
    cpu: str

    def __str__(self):
        output = 'General params:\n'
        output += f'\tUse CPU: {self.cpu}\n' 
        output += f'\tNum workers: {self.workers}\n' 
        output += f'\tPretrained weights: {self.pretrained_weights}\n' 
        output += f'\tTest model every n epochs: {self.test_model_epoch}\n'
        return output

@dataclass
class Dataparams:
    dataset_name: str
    model_in_w: int
    model_in_h: int
    box_per_split: int
    img_split_size: int

    def __str__(self):
        output = 'Dataparams:\n'
        output += f'\tDataset name: {self.dataset_name}\n' 
        output += f'\tModel input img size: ({self.model_in_w},{self.model_in_h})\n'
        output += f'\tBoxes per split: {self.box_per_split}\n'
        output += f'\tImage split size: {self.img_split_size}\n'
        return output

@dataclass
class Hyperparams:
    lr: float
    batch_size: int
    weight_decay: float
    epochs: int
    box_format: int
    iou_threshold: int
    loss_lambda_coord: float
    loss_lambda_nocoord: float
    backbone: str

    def __str__(self):
        output = 'Hyperparams:\n'
        output += f'\tEpochs: {self.epochs}\n' 
        output += f'\tLearning rate: {self.lr}\n'
        output += f'\tBath size: {self.batch_size}\n' 
        output += f'\tWeight decay: {self.weight_decay}\n' 
        output += f'\tBox format: {self.box_format}\n' 
        output += f'\tIoU threshold: {self.iou_threshold}\n' 
        output += f'\tLoss lambda coord: {self.loss_lambda_coord}\n' 
        output += f'\tLoss lambda nocoord: {self.loss_lambda_nocoord}\n' 
        output += f'\tModel backbone: {self.backbone}\n' 
        return output

class Config:
    def __init__(self, verbose: bool=True) -> None:
        args = get_training_args()
        self.hyperparams = Hyperparams(
            args.lr,
            args.batch_size,
            args.weight_decay,
            args.epochs,
            args.box_format,
            args.iou_threshold,
            args.loss_lambda_coord,
            args.loss_lambda_nocoord,
            args.backbone)

        self.dataparams = Dataparams(
            args.dataset_name,
            args.model_in_w,
            args.model_in_h,
            args.boxes_per_split,
            args.img_split_size)

        self.general = Generalparams(
            args.pretrained_weights,
            args.test_model_epoch,
            args.num_workers,
            args.cpu)

        if verbose:
            print(self.hyperparams)
            print(self.dataparams)
            print(self.general)

        return

def get_training_args():
    """
    Load the arguments for the script.
    Inputs: None
    Outputs: 
        >> args: (ArgumentParser args) Arguments passed for the script.
    """
    parser = argparse.ArgumentParser(description='Arguments for pix2pix inference.')

    # Hyperparams
    parser.add_argument( '--lr',                 type=float, default=2e-5,       help='Learning rate' )
    parser.add_argument( '--batch-size',         type=int,   default=8,          help='Batch size.' )
    parser.add_argument( '--weight-decay',       type=float, default=0.0005,     help='Weight decay for Adam optimizer' )
    parser.add_argument( '--epochs',             type=int,   default=180,        help='Total training epochs' )
    parser.add_argument( '--box-format',         type=str,   default='midpoint', help='Format of the bounding boxes "midpoint" or "corners".' )
    parser.add_argument( '--iou-threshold',      type=float, default=0.5,        help='Threshold for intersection over union (IoU).' )
    parser.add_argument( '--loss-lambda-coord',  type=float, default=5,          help='Lambda coord value for the loss function.' )
    parser.add_argument( '--loss-lambda-nocoord',type=float, default=0.5,        help='Lambda no coord value for the loss function.' )
    parser.add_argument( '--backbone',           type=str,   default='darknet19',help='Choose between resnet50 or darknet19.' )

    # Dataparams
    parser.add_argument( '--dataset-name',    type=str,   default='voc', help='Dataset name.' )
    parser.add_argument( '--model-in-h',      type=int,   default=448,   help="Height for the model's input" )
    parser.add_argument( '--model-in-w',      type=int,   default=448,   help="Height for the model's input" )
    parser.add_argument( '--boxes-per-split', type=int,   default=2,     help="Quantity of boxes per split." )
    parser.add_argument( '--img-split-size',  type=int,   default=7,     help="Size (h,w) for the image splits." )

    # General
    parser.add_argument( '--cpu',                type=bool, default=False, help="To use cpu use this to one." )
    parser.add_argument( '--num-workers',        type=int,  default=2,     help="Num workers for DataLoaders" )
    parser.add_argument( '--pretrained-weights', type=str,  default=None,  help="Path to the pretrained_weights" )
    parser.add_argument( '--test-model-epoch',   type=int,  default=5,     help="Sets the number of epochs to be trained before testing the model." )

    args = parser.parse_args()
    return args
