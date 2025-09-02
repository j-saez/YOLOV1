from typing import Dict, Tuple
from torch.utils.data import Dataset
from datasets.coco import COCODataset

def load_dataset(config: Dict, split: int) -> Tuple:
    dataset_name = config["dataset"]["name"]
    available_datasets = ["coco"]

    if (dataset_name not in available_datasets):
        raise ValueError(f"{dataset_name} is not valid. Choose from: {available_datasets}")

    dataset = Dataset()
    if dataset_name == 'coco':
        dataset = COCODataset(config, split)
        config["dataset"]["img_chs"] = dataset.get_img_chs()
        config["dataset"]["num_classes"] = dataset.get_num_classes()

    return dataset, config
