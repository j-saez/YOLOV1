from pathlib import Path
from typing import Dict
from datasets import COCODataset
from torch.utils.data import Dataset
import fiftyone.zoo as foz

AVAILABLE_DATASETS = ["coco"]

def load_dataset(config: Dict, split: int) -> Dataset:
    """
    Load a dataset based on the experiment configuration.

    This function initializes a dataset object (currently only COCO is supported),
    attaches relevant dataset metadata to the configuration, and returns both
    the dataset and the updated config.

    Args:
        config (Dict): Experiment configuration dictionary. Must contain:
            - config["dataset"]["name"]: str
                Name of the dataset to load (e.g., "coco").
            - config["dataset"]["path"]: str
                Path to the dataset files.
            - config["dataset"]["img_chs"]:
                Number of channels in the dataset images.
            - config["dataset"]["num_classes"]: int
                Number of classes present in the dataset.
        split (int): The dataset split to load (e.g., train, val, test).
            Convention is: 0 = train, 1 = val, 2 = test).

    Returns:
        Tuple[Dataset, Dict]:
            - A PyTorch Dataset object for the requested split.
    Raises:
        ValueError: If the dataset name is not supported.

    Notes:
        - Currently only supports the COCO dataset via `datasets.coco.COCODataset`.
        - Additional datasets must be added to `AVAILABLE_DATASETS` and handled here.
    """

    dataset_name = config["dataset"]["name"]
    if (dataset_name not in AVAILABLE_DATASETS):
        raise ValueError(f"{dataset_name} is not valid. Choose from: {AVAILABLE_DATASETS}")

    dataset = Dataset()
    if dataset_name == 'coco':
        dataset = COCODataset(config, split)

    return dataset

def download_dataset(config: Dict) -> None:
    """
    Download a dataset from FiftyOne Dataset Zoo (for COCO 2017) or via
    direct URL/manual methods for others.

    Args:
        config (Dict): Experiment configuration dictionary. Must contain:
            - config["dataset"]["name"]: str
                Name of the dataset ("coco" uses FiftyOne, others use manual download).
            - config["dataset"]["split"]: str, optional
                Split for datasets that support it (e.g., "train", "validation", "test").
            - config["dataset"]["path"]: str
                Local directory/file path where the dataset should be saved/exported.

    Returns:
        None

    Side Effects:
        - Downloads and saves dataset to the specified path if it does not exist.
    """
    dataset_name = config["dataset"]["name"].lower()
    split = config["dataset"].get("split", None)
    output_path = Path(config["dataset"]["path"])

    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"{dataset_name} is not valid. Choose from: {AVAILABLE_DATASETS}")

    # Skip if dataset already exists
    if output_path.exists():
        print(f"[INFO] Dataset already exists at {output_path}. Skipping download.")
        return

    if dataset_name == "coco":
        # Use FiftyOne for COCO 2017
        print(f"[INFO] Downloading COCO 2017 ({split}) via FiftyOne Dataset Zoo...")
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            dataset_name=f"coco-2017-{split}" if split else "coco-2017",
        )

        print(f"[INFO] Exporting COCO 2017 ({split}) to {output_path}...")
        dataset.export(
            export_dir=str(output_path),
            dataset_type=foz.types.COCODetectionDataset,
            split=split,
        )

        return
