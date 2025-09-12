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
        labels = self.__get_xywh_normalised_label__(label_path)
        labels = self.__convert_to_yolov1_label__(labels)

        image = Image.open(image_path).convert("RGB")

        return self.transform(image), labels


    def __get_xywh_normalised_label__(self, xml_file: str) -> torch.Tensor:
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

    def __convert_to_yolov1_label__( self, boxes_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts per-box normalized labels to YOLO v1 cell format.

        Args:
            boxes_tensor: [num_boxes, num_clases+5] tensor (one-hot classes + conf + x,y,w,h normalized to 0..1)

        Returns:
            yolov1_label: [S, S, num_boxes per cell *5 + num_clases] tensor
        """
        S = self.config["model"]["split_size"]
        B = self.config["model"]["num_boxes"]
        C = self.total_classes

        device = boxes_tensor.device
        yolov1_label = torch.zeros((S, S, B*5 + C), device=device)

        for box in boxes_tensor:
            class_one_hot = box[:C]
            conf, x, y, w, h = box[C:]

            # 1️⃣ Determine which cell the object belongs to
            cell_i = int(y * S)
            cell_j = int(x * S)
            # clamp in case x=1 or y=1
            cell_i = min(cell_i, S-1)
            cell_j = min(cell_j, S-1)

            # 2️⃣ Choose which of the B boxes to assign
            # YOLO v1 assigns the **first box** if empty; if multiple objects per cell, more complex logic can be added
            if yolov1_label[cell_i, cell_j, 4] == 0:  # check conf of first box
                box_offset = 0
            elif B > 1 and yolov1_label[cell_i, cell_j, 9] == 0:  # second box
                box_offset = 5
            else:
                # Already full, skip or override (depends on your strategy)
                continue

            # 3️⃣ Fill in box info relative to cell
            x_cell = x * S - cell_j  # relative x within cell (0..1)
            y_cell = y * S - cell_i  # relative y within cell (0..1)

            yolov1_label[cell_i, cell_j, box_offset:box_offset+5] = torch.tensor([x_cell, y_cell, w, h, 1.0], device=device)
            # 4️⃣ Fill class probabilities (YOLO v1 uses same class vector for both boxes in cell)
            yolov1_label[cell_i, cell_j, B*5:] = class_one_hot

        return yolov1_label
