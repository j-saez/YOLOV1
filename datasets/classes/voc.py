import os
import torch
import torchvision
import pandas   as pd
import torch.nn as nn
from PIL              import Image
from torch.utils.data import Dataset

IMGS_CHS = 3
IMGS_IDX = 0
LABELS_IDX = 1
DATASET_DIR = f'{os.getcwd()}/datasets/data/voc'
IMAGES_DIR = f'{DATASET_DIR}/images'
LABELS_DIR = f'{DATASET_DIR}/labels'
TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize([0.5 for _ in range(IMGS_CHS)],[0.5 for _ in range(IMGS_CHS)])])

class VOCDataset(Dataset):
    def __init__(self,data_split: str, img_split_size: int, box_per_split: int, model_in_w: int, model_in_h: int):
        """
        Class that loads the VOCDataset
        Inputs:
            >> data_split: (str) Indicates whether to load the training or test data. ["train" or "test"]
            >> split_size: (int) Indicates the size of the output grids from the model.
            >> box_per_split: (int) Indicates the quantity of boxes that the model will output per split.
            >> model_in_w: (int) Indicates the width of the image for the model
            >> model_in_h: (int) Indicates the height of the image for the model

        Attributes:
            >> csv_data: (pd.DataFrame) Pandas dataframe containing the information from the csv file.
            >> S: (int) Indicates the size of the output grids from the model.
            >> B: (int) Quantity of box per split.
            >> C: (int) Total quantity of classes.
            >> model_in_w: (int) Indicates the width of the image for the model
            >> model_in_h: (int) Indicates the height of the image for the model

        """
        csv_filename = f'{DATASET_DIR}/train.csv' if data_split == 'train' else f'{DATASET_DIR}/test.csv'
        self.csv_data = pd.read_csv(csv_filename)
        self.S = img_split_size
        self.B = box_per_split
        self.C = 20 # this dataset contains 20 classes
        self.model_in_w = model_in_w
        self.model_in_h = model_in_h
        return

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        """
        Returns the idxth item of the dataset.
        Inputs:
            >> idx: (int) Idx of the data to be loaded.
        Outputs:
            >> image: (torch.tensor of size [chs, model_in_w, model_in_h]) Idxth image of the dataset
            >> labels: (torch tensor [S, S, C+len(boxes)*5], where C+5 contains [c1, ..., cC, p, x, y, w, h]) Label for the idxth image of the dataset.
        """
        img_filename = f'{IMAGES_DIR}/{self.csv_data.iloc[idx, IMGS_IDX]}'
        label_filename = f'{LABELS_DIR}/{self.csv_data.iloc[idx, LABELS_IDX]}'

        # Load boxes from label_{}.txt
        boxes = []
        with open(label_filename, 'r') as f:
            for line in f.readlines():
                box = line[:-1].split(' ')
                boxes.append([float(item) for item in box])

        # Image to [-1, 1] tensor
        img = Image.open(img_filename).resize((self.model_in_w, self.model_in_h))
        img = TRANSFORMS(img)

        # Create labels from the boxes
        label = self.__create_label__(boxes)
        return img, label

    def __create_label__(self, boxes: list):
        """
        Creates labels in with shape: [c1, ..., C, p, x, y, w, h], where:
            - cx indicates the class in one-hot encoding,
            - p indicates if there is an object (1) or not (0),
            - x indicates x coord of the box's centroid
            - y indicates y coord of the box's centroid
            - h indicates height of the box
            - w indicates width of the box

        Inputs:
            >> boxes: (list of list) In each poistion of the list, contains another list with [int label, x, y, w, h]. THIS VALUES ARE RELATIVE TO THE IMAGE
        Outputs:
            >> labels: (torch tensor [S, S, C+5]), where C+5 contains [c1, ..., cC, p, x, y, w, h]
        """

        labels = torch.zeros(self.S, self.S, self.C+(5*self.B))
        for box in boxes:
            # Values relative to the image
            class_label, x, y, w, h = box
            
            # Get the cell where the box is located: x,y in [0,1] -> get the cell where x,y is located in the S,S splitted image.
            row, column = int(self.S * y), int(self.S * x)

            # Get locations relative to the cell
            x_cell, y_cell = self.S * x - column, self.S * y - row 
            w_cell, h_cell = self.S * w, self.S * h 

            if labels[row,column,self.C] == 0:
                one_hot_label = nn.functional.one_hot(torch.tensor(int(class_label)), self.C)
                cell_box = torch.tensor([1, x_cell, y_cell, w_cell, h_cell])
                # Get a label with size (B,S,S,C+(5*B)) instead of (B,S,S,C+5) so that the get_bbox works 
                label = torch.cat(
                    dim=0,
                    tensors=(one_hot_label, cell_box, torch.zeros(5)))
                labels[row,column] = label
        return labels
