import torch
import argparse
from dataclasses import dataclass

@dataclass
class Dataparams:
    dataset_name: str
    model_in_w: int
    model_in_h: int

    def __str__(self):
        output = 'Dataparams:\n'
        output += f'\tDataset name: {self.dataset_name}\n' 
        output += f'\tModel input img size: ({self.model_in_w},{self.model_in_h})\n'
        return output

@dataclass
class Hyperparams:
    lr: float
    batch_size: int
    weight_decay: float
    epochs: int

    def __str__(self):
        output = 'Hyperparams:\n'
        output += f'\tEpochs: {self.epochs}\n' 
        output += f'\tLearning rate: {self.lr}\n'
        output += f'\tBath size: {self.batch_size}\n' 
        output += f'\tWeight decay: {self.weight_decay}\n' 
        return output

class Config:
    def __init__(self, verbose: bool=True) -> None:
        args = get_training_args()
        self.hyperparams = Hyperparams(
            args.lr,
            args.batch_size,
            args.weight_decay,
            args.epochs)
        if verbose:
            print(self.hyperparams)
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
    parser.add_argument( '--lr',           type=float, default=2e-5, help='Learning rate' )
    parser.add_argument( '--batch-size',   type=int,   default=16,   help='Batch size.' )
    parser.add_argument( '--weight-decay', type=float, default=0.0,  help='Weight decay for Adam optimizer' )
    parser.add_argument( '--epochs',       type=int,   default=1000, help='Total training epochs' )

    # Dataparams
    parser.add_argument( '--dataset-name', type=str,   default='voc', help='Dataset name.' )
    parser.add_argument( '--input-img-h',  type=int,   default=448,   help="Height for the model's input" )
    parser.add_argument( '--input-img-w',  type=int,   default=448,   help="Height for the model's input" )

    args = parser.parse_args()
    return args
