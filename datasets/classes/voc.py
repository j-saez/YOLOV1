import torch
import torchvision
from torch.utils.data import Dataset
from typing import Dict

class VOCDataset(Dataset):
    def __init__(self, config: Dict, split: int):
        """
        Class that loads the VOCDataset
        Inputs:
            >> config: (Dict) TODO add config information regarding the dataset (fields and so on...)
            >> split (int): The dataset split to load (e.g., train, val, test).
                Convention is: 0 = train, 1 = val, 2 = test).

        Attributes: TODO

        """
        self.config = config

        from .utils import DatasetSplitEnum
        if (split > DatasetSplitEnum.TEST.value):
            raise ValueError(
                f"Split value is {split}, but valid values are {[e.value for e in DatasetSplitEnum]} "
                f"({[e.name for e in DatasetSplitEnum]})"
            )

        # TODO: Define transformations for train, val and test for the voc dataset
        self.transform = torchvision.transforms.Compose([])

        return

    def __len__(self):
        """
        TODO
        """
        return -1

    def __getitem__(self, idx: int):
        """
        TODO
        Returns the idxth item of the dataset.
        Inputs:
            >> idx: (int) Idx of the data to be loaded.
        Outputs:
            >> images: (torch.tensor of size [chs, model_in_w, model_in_h]) Idxth image of the dataset
            >> labels: (torch tensor [S, S, C+len(boxes)*5], where C+5 contains [c1, ..., cC, p, x, y, w, h]) Label for the idxth image of the dataset.
        """
        images = torch.Tensor([])
        labels = torch.Tensor([])
        return images, labels
