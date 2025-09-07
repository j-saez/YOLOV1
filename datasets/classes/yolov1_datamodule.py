import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.classes import load_dataset, download_dataset
from typing import Dict
from datasets.classes.utils import DatasetSplitEnum

class YOLOV1DataModule(pl.LightningDataModule):

    def __init__(self, config: Dict) -> None:
        super().__init__()

        # TODO: Research a bit more what prepare_data_per_node does.
        self.prepare_data_per_node = config["dataset"]["prepare_per_node"]
        assert type(self.prepare_data_per_node) == bool
        self.config = config

        return

    def prepare_data(self) -> None:
        """
        Downloads the data so we have it to disc
        """
        # this is for single gpu
        # TODO: Research a litte bit more about this.
        download_dataset(self.config)
        return

    def setup(self, stage: str) -> None:
        """
        Loads the data downloaded in prepate_data as a pytorch dataset class object
        """
        # this is for multiple gpu as it is called in every gpu on the system.

        if stage == 'fit':
            self.train_dataset = load_dataset(self.config, DatasetSplitEnum.TRAIN.value)
            self.val_dataset = load_dataset(self.config, DatasetSplitEnum.VAL.value)
        elif stage == 'test':
            self.test_dataset = load_dataset(self.config, DatasetSplitEnum.TEST.value)

        return

    def train_dataloader(self,):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            pin_memory=True, # This lets .to(device, non_blocking=True) use page-locked memory → faster async transfers.
            shuffle=False
        )

    def val_dataloader(self,):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            shuffle=False
        )

    def test_dataloader(self,):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            shuffle=False
        )
