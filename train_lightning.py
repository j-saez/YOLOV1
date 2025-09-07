import os
import argparse
import pytorch_lightning as pl
from models import YOLOV1
import datasets.classes as datasets
import json
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def load_config(file_path: str) -> dict:
    """
    Loads a JSON configuration file and returns it as a dictionary.

    Args:
        file_path (str): Full path to the JSON configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the file extension is not .json
    """
    if not file_path.endswith(".json"):
        raise ValueError(f"Expected a .json file, got: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for YOLOv1 training.')
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Full path to the configuration file.'
    )
    args = parser.parse_args()

    print(f'[INFO] Loading config...')
    config = load_config(args.config_file)
    print(f'[INFO] Done.')

    print(f'[INFO] Loading DataModule...')
    data_module = datasets.YOLOV1DataModule(config)
    print(f'[INFO] Done.')

    print(f'[INFO] Loading YOLOV1 model...')
    model = pl.LightningModule()
    if config["training"]["from_pretrained"] == True:
        print(f'[INFO]      Loading the model from the following pretrained weights: {config["training"]["weights_path"]}')
        model = YOLOV1.load_from_checkpoint(config["training"]["weights_path"])
    else:
        print(f'[INFO]      Training the model from scratch.')
        model = YOLOV1(config)
    print(f'[INFO] Done.')

    print(f'[INFO] Running healtchecks...')
    print(f'[INFO] HEALTHCHECK HAVE NOT BEEN IMPLEMENTED YET')
    print(f'*************************************************')

    checkpoint_filename = f'YOLOV1_{config["dataset"]["name"]}_{os.path.basename(args.config_file)}Config'
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{os.getcwd()}/runs/checkpoints',
        filename=checkpoint_filename+'_e{epoch}_mClsAcc{val_acc:.2f}',
        save_top_k = 3,
        monitor='val_yolov1_loss',
        mode='min',
        verbose=True,
        save_on_train_epoch_end=False # Save after validation
    )

    tb_logger = TensorBoardLogger(
        save_dir='runs/tensorboard',
        name='YOLOV1',
        version=f'{config["dataset"]["name"]}_{os.path.basename(args.config_file)}Config',
        default_hp_metric=False
    )

    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        logger=tb_logger,
        accelerator=config["accelerator"],
        devices=config["gpus_ids"],
        max_epochs=config["training"]["epochs"],
        log_every_n_steps=config["training"]["logging_steps"]
    )

    print(f'[INFO] Starting of the training process...')
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
