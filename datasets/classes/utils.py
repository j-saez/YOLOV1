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
    - COCO via kagglehub (train, val, test).

    Args:
        config (Dict): Experiment configuration dictionary.

    Returns:
        None
    """
    dataset_name = config["dataset"]["name"].lower()
    output_path = Path(config["dataset"]["path"])

    if dataset_name not in ["voc", "coco"]:
        raise ValueError(f"{dataset_name} is not valid. Choose from: ['voc', 'coco']")

    # Ask the user to create dataset path if it does not exist
    if not output_path.exists():
        response = input(
            f"[WARNING] Dataset path {output_path} does not exist. Create it? [y/N]: "
        ).strip().lower()
        if response == "y":
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created {output_path}")
        else:
            print("[ERROR] Dataset path does not exist. Aborting.")
            return

    # Dataset-specific existence check
    if dataset_name == "voc":
        voc_dir = output_path / "VOC_dataset"
        if voc_dir.exists():
            print(f"[INFO] VOC dataset already exists at {voc_dir}. Download aborted.")
            return

        print("[INFO] Downloading the VOC Dataset...")
        temp_path = Path(kagglehub.dataset_download("bardiaardakanian/voc0712"))
        print("[INFO] VOC Dataset downloaded to:", temp_path)
        shutil.move(str(temp_path), voc_dir)
        print("[INFO] VOC Dataset moved to:", voc_dir)

    elif dataset_name == "coco":
        coco_dir = output_path / "COCO_dataset"
        if coco_dir.exists():
            print(f"[INFO] COCO dataset already exists at {coco_dir}. Download aborted.")
            return

        print("[INFO] Downloading the COCO 2017 Dataset...")
        temp_path = Path(kagglehub.dataset_download("awsaf49/coco-2017-dataset"))
        print("[INFO] COCO 2017 Dataset downloaded to:", temp_path)
        shutil.move(str(temp_path), coco_dir)
        print("[INFO] COCO Dataset moved to:", coco_dir)

    return
