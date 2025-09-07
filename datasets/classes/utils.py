import os
import kagglehub
import shutil
from pathlib import Path
from typing import Dict
from datasets.classes.coco import COCODataset
from datasets.classes.voc import VOCDataset
from torch.utils.data import Dataset
from enum import Enum

class DatasetSplitEnum(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

AVAILABLE_DATASETS = ["coco", "voc"]

def load_dataset(config: Dict, split: int) -> Dataset:
    """
    Load a dataset based on the experiment configuration.

    Args:
        config (Dict): Experiment configuration dictionary.
        split (int): The dataset split to load (0 = train, 1 = val, 2 = test).

    Returns:
        Dataset: A PyTorch Dataset object for the requested split.
    """
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"{dataset_name} is not valid. Choose from: {AVAILABLE_DATASETS}")

    if dataset_name == "coco":
        return COCODataset(config, split)
    elif dataset_name == "voc":
        return VOCDataset(config, split)

    raise ValueError(f"Dataset {dataset_name} not supported")

def download_dataset(config: Dict) -> None:
    """
    Download a dataset:
    - VOC via kagglehub (all valid splits for the year, extracted locally).
    - COCO via FiftyOne (train, val, test).

    Args:
        config (Dict): Experiment configuration dictionary.

    Returns:
        None
    """
    dataset_name = config["dataset"]["name"].lower()
    output_path = Path(config["dataset"]["path"])

    if dataset_name not in ["voc", "coco"]:
        raise ValueError(f"{dataset_name} is not valid. Choose from: ['voc', 'coco']")

    if output_path.exists():
        f"[INFO] {dataset_name} is already present. Download aborted."
        return

    if not output_path.parent.exists():
        response = input(
            f"[WARNING] Parent directory {output_path.parent} does not exist. Create it? [y/N]: "
        ).strip().lower()
        if response == "y":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created {output_path.parent}")
        else:
            print("[ERROR] Dataset path does not exist. Aborting.")
            return

    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "voc":
        print("[INFO] Downloading the VOC Dataset...")
        temp_path = Path(kagglehub.dataset_download("bardiaardakanian/voc0712"))
        print("[INFO] VOC Dataset downloaded to:", temp_path)
        for item in temp_path.iterdir():
            shutil.move(str(item), output_path)
        print("[INFO] VOC Dataset moved to:", output_path)

    elif dataset_name == "coco":
        print("[INFO] Downloading the COCO 2017 Dataset...")
        temp_path = Path(kagglehub.dataset_download("awsaf49/coco-2017-dataset"))
        print("[INFO] COCO 2017 Dataset downloaded to:", temp_path)
        for item in temp_path.iterdir():
            shutil.move(str(item), output_path)
        print("[INFO] COCO 2017 Dataset moved to:", output_path)
