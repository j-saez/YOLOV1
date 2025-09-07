import torch
import torch.nn as nn
import pytorch_lightning as torch_lightning

from typing import Dict
from models import backbones
from training.loss import YOLOV1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class YOLOV1(torch_lightning.LightningModule):

    def __init__( self, conf: Dict):
        super().__init__()
        self.conf = conf

        split_size = self.conf["model"]["split_size"]
        num_boxes = self.conf["model"]["num_boxes"]
        num_classes = self.conf["dataset"]["num_classes"]
        data_per_box = 5 # The 5 values are: (prob,x,y,w,h)

        backbone, backbone_out_feat = backbones.load(conf)


        fcl = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_feat * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # (S,S,30) where (num_classes + num_boxes * 5) = 30, and 5 is for (prob,x,y,w,h)
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * data_per_box))
        )

        self.model = nn.Sequential(backbone, fcl)
        self.loss_function = YOLOV1Loss(conf)
        self.metric = MeanAveragePrecision()

        return

    def training_step(self, images: torch.Tensor, labels: torch.Tensor, batch_idx: int):
        device = next(self.model.parameters()).device  # guaranteed to be a torch.device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            preds = self.model(images)
            loss_list = self.loss_function(preds,labels)
            [
                yolov1_loss,
                box_loss,
                object_loss,
                noobject_loss,
                prob_loss
            ] = loss_list

            # on_step = True --> Logs the metric at the current step
            # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
            self.log_dict({
                f"train_yolov1_loss": yolov1_loss,
                f"train_box_loss": box_loss,
                f"train_object_loss": object_loss,
                f"train_no_object_loss": noobject_loss,
                f"train_prob_loss": prob_loss,
            }, on_step=False, on_epoch=True)
        return

    def validation_step(self, images: torch.Tensor, labels: torch.Tensor, batch_idx: int):
        device = next(self.model.parameters()).device  # guaranteed to be a torch.device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            preds = self.model(images)
            loss_list = self.loss_function(preds,labels)
            [
                yolov1_loss,
                box_loss,
                object_loss,
                noobject_loss,
                prob_loss
            ] = loss_list

            # on_step = True --> Logs the metric at the current step
            # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
            self.log_dict({
                f"val_yolov1_loss": yolov1_loss,
                f"val_box_loss": box_loss,
                f"val_object_loss": object_loss,
                f"val_no_object_loss": noobject_loss,
                f"val_prob_loss": prob_loss,
            }, on_step=False, on_epoch=True)

            self.metric.update(
                self.to_torchmetrics_format(preds),
                self.to_torchmetrics_format(labels),
            )
        return

    def test_step(self, images: torch.Tensor, labels: torch.Tensor, batch_idx: int):
        device = next(self.model.parameters()).device  # guaranteed to be a torch.device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            preds = self.model(images)
            loss_list = self.loss_function(preds,labels)
            [
                yolov1_loss,
                box_loss,
                object_loss,
                noobject_loss,
                prob_loss
            ] = loss_list

            # on_step = True --> Logs the metric at the current step
            # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
            self.log_dict({
                f"test_yolov1_loss": yolov1_loss,
                f"test_box_loss": box_loss,
                f"test_object_loss": object_loss,
                f"test_no_object_loss": noobject_loss,
                f"test_prob_loss": prob_loss,
            }, on_step=False, on_epoch=True)

            self.metric.update(
                self.to_torchmetrics_format(preds),
                self.to_torchmetrics_format(labels),
            )
        return

    def to_torchmetrics_format(self, preds: torch.Tensor):
        """
        Converts YOLOv1 predictions or labels into the format required by TorchMetrics MeanAveragePrecision.

        Parameters
        ----------
        preds : torch.Tensor
            Tensor of shape [batch_size, S*S*(C + B*5)] or [batch_size, S, S, C + B*5].
            Contains class probabilities, bounding box coordinates, and objectness scores.

        Returns
        -------
        list[dict]
            A list of length batch_size, where each element is a dict with:
            - "boxes": tensor of shape [num_boxes, 4] in (x1, y1, x2, y2) format
            - "scores": tensor of shape [num_boxes]
            - "labels": tensor of shape [num_boxes]
        """
        batch_size = preds.shape[0]
        S = self.conf["model"]["split_size"]
        B = self.conf["model"]["num_boxes"]
        C = self.conf["dataset"]["num_classes"]

        results = []

        # Reshape to [batch, S, S, C + B*5]
        preds = preds.view(batch_size, S, S, C + B*5)

        for b in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []

            for i in range(S):
                for j in range(S):
                    cell = preds[b, i, j]
                    class_probs = cell[:C]  # [C]
                    for k in range(B):
                        # Bounding box: [p, x, y, w, h]
                        offset = C + k*5
                        p = cell[offset]
                        x = (cell[offset+1] + j) / S  # normalize to [0,1] relative to image
                        y = (cell[offset+2] + i) / S
                        w = cell[offset+3]
                        h = cell[offset+4]

                        if p > 0:  # only consider boxes with nonzero confidence
                            # Convert to x1,y1,x2,y2
                            x1 = x - w/2
                            y1 = y - h/2
                            x2 = x + w/2
                            y2 = y + h/2
                            # Class label is argmax
                            cls = class_probs.argmax()
                            boxes_list.append([x1, y1, x2, y2])
                            scores_list.append(p)
                            labels_list.append(cls)

            if len(boxes_list) > 0:
                results.append({
                    "boxes": torch.tensor(boxes_list, device=preds.device),
                    "scores": torch.tensor(scores_list, device=preds.device),
                    "labels": torch.tensor(labels_list, device=preds.device, dtype=torch.int64)
                })
            else:
                # If no objects detected, return empty tensors
                results.append({
                    "boxes": torch.empty((0,4), device=preds.device),
                    "scores": torch.empty((0,), device=preds.device),
                    "labels": torch.empty((0,), device=preds.device, dtype=torch.int64)
                })
        return results

    def on_validation_epoch_end(self):
        """
        Computes and logs evaluation metrics at the end of a validation epoch.

        This method performs the following actions:
            0. Uses automatic mixed precision (AMP) and disables gradient computation
               for efficient evaluation on GPU.
            1. Computes metrics using `self.metric.compute()`.
            2. Logs key metrics:
                - "val/mAP"   : mean Average Precision over all IoU thresholds.
                - "val/mAP49" : mAP at IoU=0.5.
                - "val/Recall": mean recall for top 99 detections (mar_100).
            3. Resets the metric state to prepare for the next epoch.

        Notes
        -----
        - This method assumes `self.metric` has `compute()` and `reset()` methods,
          typically from a PyTorch Lightning or TorchMetrics metric object.
        - Logging is done using `self.log()`, which is compatible with PyTorch Lightning.

        Returns
        -------
        None
        """
        device = next(self.model.parameters()).device  # guaranteed to be a torch.device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            results = self.metric.compute()
            self.log("val/mAP", results["map"])
            self.log("val/mAP50", results["map_50"])
            self.log("val/Recall", results["mar_100"])

            # Reset for next epoch
            self.metric.reset()
        return

    def on_test_epoch_end(self):
        """
        Computes and logs evaluation metrics at the end of a testing epoch.

        This method performs the following actions:
            1. Uses automatic mixed precision (AMP) and disables gradient computation
               for efficient evaluation on GPU.
            2. Computes metrics using `self.metric.compute()`.
            3. Logs key metrics:
                - "test/mAP"   : mean Average Precision over all IoU thresholds.
                - "test/mAP50" : mAP at IoU=0.5.
                - "test/Recall": mean recall for top 100 detections (mar_100).
            4. Resets the metric state to prepare for the next epoch.

        Notes
        -----
        - This method assumes `self.metric` has `compute()` and `reset()` methods,
          typically from a PyTorch Lightning or TorchMetrics metric object.
        - Logging is done using `self.log()`, which is compatible with PyTorch Lightning.

        Returns
        -------
        None
        """
        device = next(self.model.parameters()).device  # guaranteed to be a torch.device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            results = self.metric.compute()
            self.log("test/mAP", results["map"])
            self.log("test/mAP50", results["map_50"])
            self.log("test/Recall", results["mar_100"])

            # Reset for next epoch
            self.metric.reset()
        return

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        This method performs the following steps:
            1. Validates the optimizer hyperparameters using `__optim_hyperaprams_healthcheck__`.
            2. Creates an optimizer (Adam or AdamW) based on the configuration.
            3. Validates the scheduler hyperparameters using `__scheduler_hyperaprams_healthcheck__`.
            4. Creates a learning rate scheduler (StepLR or CosineAnnealingLR) if specified in the config.

        Optimizer configuration is read from:
            self.conf["training"]["hyperparams"]["optim"]

        Scheduler configuration is read from:
            self.conf["training"]["hyperparams"]["lr_scheduler"]

        Returns
        -------
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler] | None
            - A list containing the optimizer and a list containing the scheduler.
            - If no scheduler is configured (`sched_name` is None), only the optimizer is returned.

        Raises
        ------
        KeyError
            - If required optimizer or scheduler parameters are missing or invalid,
              as checked by the respective health check methods.

        Notes
        -----
        - Supports the following optimizers:
            - 'adam'
            - 'adamw'
        - Supports the following schedulers:
            - 'StepLR' (requires step_size, gamma, last_epoch)
            - 'CosineAnnealingLR' (requires eta_min, last_epoch)
        - The method ensures hyperparameters are valid before creating the optimizer
          and scheduler objects.
        """
        self.__optim_hyperaprams_healthcheck__()

        optim_conf = self.conf["training"]["hyperparams"]["optim"]
        optimizer_classes = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW
        }

        optimizer_class = optimizer_classes[optim_conf["name"]]
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=optim_conf["learning_rate"],
            weight_decay=optim_conf["weight_decay"],
            betas=(
                optim_conf["adam_beta1"],
                optim_conf["adam_beta2"]
            )
        )

        self.__scheduler_hyperaprams_healthcheck__()

        scheduler = None
        sched_conf = self.conf["training"]["hyperparams"]["lr_scheduler"]
        sched_name = sched_conf["name"]

        if sched_name == None:
            return optimizer

        if sched_name == "cosineannealinglr":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.conf["training"]["epochs"],
                eta_min=sched_conf.get("eta_min", 1e-5),
                last_epoch=sched_conf.get("last_epoch", -1)
            )
        elif sched_name == "cosineannealinglr":
            scheduler = StepLR(
                optimizer,
                step_size=sched_conf["step_size"],
                gamma=sched_conf["gamma"],
                last_epoch=sched_conf["last_epoch"]
            )

        return [optimizer], [scheduler]

    def __optim_hyperaprams_healthcheck__(self) -> None:
        """
        Validates the optimizer configuration in the training hyperparameters.

        This function checks that the optimizer section of the configuration
        (`self.conf["training"]["hyperparams"]["optim"]`) is correctly defined.
        It ensures that the optimizer name is valid and that all required parameters
        for the selected optimizer are present.

        Supported optimizers and required parameters:
            - 'adam'
            - 'adamw'

        Required parameters for both optimizers:
            - learning_rate
            - weights_decay
            - adam_beta1
            - adam_beta2

        Raises
        ------
        KeyError
            - If the optimizer name is missing from the configuration.
            - If the optimizer name is invalid (not 'adam' or 'adamw').
            - If any required parameter (learning_rate, weights_decay, adam_beta1, adam_beta2) is missing.

        Returns
        -------
        None
            If all checks pass, the function returns nothing.
        """
        try:
            _ = self.conf["training"]["hyperparams"]["optim"]["name"]
        except KeyError:
            raise KeyError("optim name not defined in self.conf['hyperparams']['optim']")

        optim_name = self.conf["training"]["hyperparams"]["optim"]["name"]
        if (optim_name not in ['adam', 'adamw']):
            raise KeyError(f"optim name ({optim_name}) is not valid. Choose between 'adam' or 'adamw'.")

        try:
            print(self.conf)
            _ = self.conf["training"]["hyperparams"]["optim"]["learning_rate"]
        except KeyError:
            raise KeyError("learning_rate not defined in ['hyperparams']['optim']")

        try:
            _ = self.conf["training"]["hyperparams"]["optim"]["weight_decay"]
        except KeyError:
            raise KeyError("weight_decay not defined in ['hyperparams']['optim']")

        try:
            _ = self.conf["training"]["hyperparams"]["optim"]["adam_beta1"]
        except KeyError:
            raise KeyError("adam_beta1 not defined in ['hyperparams']['optim']")

        try:
            _ = self.conf["training"]["hyperparams"]["optim"]["adam_beta2"]
        except KeyError:
            raise KeyError("adam_beta2 not defined in ['hyperparams']['optim']")

        return

    def __scheduler_hyperaprams_healthcheck__(self):
        """
        Validates the scheduler configuration in the training hyperparameters.

        This function checks that the scheduler section of the configuration
        (`self.conf["training"]["hyperparams"]["lr_scheduler"]`) is correctly defined.
        It ensures that the scheduler name is valid and that all required parameters
        for the selected scheduler are present.

        Supported schedulers and required parameters:
            - None: no scheduler is used.
            - 'StepLR':
                - step_size
                - gamma
                - last_epoch
            - 'CosineAnnealingLR':
                - eta_min
                - last_epoch

        Raises
        ------
        KeyError
            - If the scheduler name is missing from the configuration.
            - If the scheduler name is invalid (not None, 'StepLR', or 'CosineAnnealingLR').
            - If required parameters for the selected scheduler are missing.

        Returns
        -------
        None
            If all checks pass, the function returns nothing.
        """

        try:
            _ = self.conf["training"]["hyperparams"]["lr_scheduler"]["name"]
        except KeyError:
            raise KeyError("lr_scheduler name not defined in ['hyperparams']['lr_scheduler']")

        lr_scheduler_config = self.conf["training"]["hyperparams"]["lr_scheduler"]
        if lr_scheduler_config["name"] == None:
            return

        elif lr_scheduler_config["name"] not in ['steplr','cosineannealinglr']:
            raise KeyError("Invalid name for the lr_scheduler. Valids are: None, 'steplr','cosineannealinglr'.")

        params = lr_scheduler_config["params"]
        if lr_scheduler_config["name"] == 'steplr':
            try:
                _ = params['step_size']
            except KeyError:
                raise KeyError("The StepLR schduler must contain the step_size and it is not in the config file.")

            try:
                _ = params['gamma']
            except KeyError:
                raise KeyError("The StepLR schduler must contain the gamma value and it is not in the config file.")

            try:
                _ = params['last_epoch']
            except KeyError:
                raise KeyError("The StepLR schduler must contain the last_epoch value and it is not in the config file.")

        elif lr_scheduler_config["name"] == 'cosineannealinglr':
            try:
                _ = params['eta_min']
            except KeyError:
                raise KeyError("The CosineAnnealingLR schduler must contain the eta_min and it is not in the config file.")

            try:
                _ = params['last_epoch']
            except KeyError:
                raise KeyError("The CosineAnnealingLR schduler must contain the epoch_value value and it is not in the config file.")

        return
