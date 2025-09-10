import torch
import torchvision
import xml.etree.ElementTree as ET
import torchvision.transforms as T

from torch.utils.data import Dataset
from typing import Dict, Tuple
from PIL import Image


class VOCDataset(Dataset):
    def __init__(self, config: Dict, split: int):
        """
        Initialize the VOC2007 dataset loader.

        This class loads images and annotations from the VOC2007 dataset, applies
        transformations, and prepares labels in a format suitable for YOLOv1 training.

        Parameters
        ----------
        config : dict
            Experiment configuration dictionary containing dataset and model parameters.
            Expected keys:
                - dataset["path"]: str, path to the dataset root
                - model["input_w"]: int, width of the input image
                - model["input_h"]: int, height of the input image
        split : int
            Dataset split to load:
                0 = train
                1 = val
                2 = test

        Attributes
        ----------
        total_classes : int
            Total number of classes in VOC2007 (20).
        prob_x_y_w_h : int
            Number of values per bounding box (x_center, y_center, width, height, confidence).
        VOC_CLASS_TO_IDX : dict
            Mapping from VOC class names to integer indices.
        path : str
            Path to the VOC2007 images and annotations for the specified split.
        image_ids : list of str
            List of image IDs for the selected split.
        transform : torchvision.transforms.Compose
            Transformations applied to images. Includes resizing, flipping, color jitter (train), and conversion to tensor.
        """
        VOC_CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.total_classes = len(VOC_CLASSES)
        self.prob_x_y_w_h = 5
        self.VOC_CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
        self.config = config

        from .utils import DatasetSplitEnum
        if (split > DatasetSplitEnum.TEST.value):
            raise ValueError(
                f"Split value is {split}, but valid values are {[e.value for e in DatasetSplitEnum]} "
                f"({[e.name for e in DatasetSplitEnum]})"
            )

        set_file_name = ''
        if split == DatasetSplitEnum.TRAIN: set_file_name = 'train.txt'
        elif split == DatasetSplitEnum.VAL: set_file_name = 'val.txt'
        else: set_file_name = 'test.txt'

        self.path = config["dataset"]["path"] + '/VOC_dataset/VOC_dataset/VOCdevkit/VOC2007'
        set_file_2007 = self.path + '/ImageSets/Layout/' + set_file_name

        self.image_ids = []
        try:
            with open(set_file_2007, "r") as f:
                ids = [line.strip() for line in f.readlines()]
                self.image_ids.extend(ids)
        except FileNotFoundError:
            print(f"[WARNING] {set_file_2007} not found, skipping.")

        self.transform = torchvision.transforms.Compose([])
        if split == DatasetSplitEnum.TRAIN:
            self.transform = T.Compose([
                T.Resize((config["model"]["input_w"], config["model"]["input_h"])),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.Resize((config["model"]["input_w"], config["model"]["input_h"])),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the idx-th item of the dataset.
        """
        image_id = self.image_ids[idx]
        image_path = self.path + '/JPEGImages/' + image_id + '.jpg'

        label_path = self.path + '/Annotations/' + image_id + '.xml'
        norm_labels = self.__get_normalised_label__(label_path)

        image = Image.open(image_path).convert("RGB")

        return self.transform(image), norm_labels


    def __get_normalised_label__(self, xml_file: str) -> torch.Tensor:
        """
        Parses a VOC XML file and returns normalized bounding boxes.

        Args:
            xml_file (str or Path): Path to VOC XML annotation file.

        Returns:
            labels (torch.Tensor): Shape (num_boxes, total_classes + 5),
                                   containing one-hot class + [conf, x_center, y_center, w, h]
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image size
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        # Extract objects and normalize bounding boxes
        num_boxes = len(root.findall("object"))
        labels = torch.zeros((num_boxes, self.total_classes + self.prob_x_y_w_h))
        for i, obj in enumerate(root.findall("object")):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text) / width
            ymin = int(bbox.find("ymin").text) / height
            xmax = int(bbox.find("xmax").text) / width
            ymax = int(bbox.find("ymax").text) / height

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            box_width = xmax - xmin
            box_height = ymax - ymin
            prob = 1.0

            class_idx = self.VOC_CLASS_TO_IDX[class_name]
            labels[i][class_idx] = 1
            labels[i][-5:] = torch.tensor([prob, x_center, y_center, box_width, box_height],dtype=torch.float32)

        return labels

