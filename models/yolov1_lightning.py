import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn as nn
import pytorch_lightning as torch_lightning
from models import backbones
from training.loss import YOLOV1Loss
from typing import Dict

class YOLOV1(torch_lightning.LightningModule):

    def __init__( self, conf: Dict):
        super().__init__()
        self.conf = conf

        split_size = self.conf["model"]["split_size"]
        num_boxes = self.conf["model"]["num_boxes"]
        num_classes = self.conf["dataset"]["num_classes"]
        data_per_box = 5 # The 5 values are: (prob,x,y,w,h)

        backbone, backbone_out_feat = backbones.load(
            self.conf["model"]["backbone"],
            self.conf["dataset"]["data_chs"],
        )


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
        with torch.cuda.amp.autocast():
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
        with torch.cuda.amp.autocast(), torch.no_grad():
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
        with torch.cuda.amp.autocast(), torch.no_grad():
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

    def on_validation_epoch_end(self):
        with torch.cuda.amp.autocast(), torch.no_grad():
            results = self.metric.compute()
            self.log("val/mAP", results["map"])
            self.log("val/mAP50", results["map_50"])
            self.log("val/Recall", results["mar_100"])

            # Reset for next epoch
            self.metric.reset()
        return

    def on_test_epoch_end(self):
        with torch.cuda.amp.autocast(), torch.no_grad():
            results = self.metric.compute()
            self.log("test/mAP", results["map"])
            self.log("test/mAP50", results["map_50"])
            self.log("test/Recall", results["mar_100"])

            # Reset for next epoch
            self.metric.reset()
        return

    def configure_optimizers(self):
        """
        Configures the optimizer that will be used during the training process
        Inputs: None
        Outputs:
            >> optimizers_list: (list) Contains the optimizers for the generator, discriminator and classifier.
            >> lr_schedulers_list: (list) Contains the lr schedulers for the generator, discriminator and classifier.
        """
        yolov1_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf["hyperparams"]["lr"],
            weight_decay=self.conf["hyperparams"]["wights_decay"],
            betas=(
                self.conf["hyperams"]["optim"]["beta1"],
                self.conf["hyperams"]["optim"]["beta2"]
            )
        )

        # TODO: Check the yolov1 lr scheduler to be used
        #yolov1_lr_scheduler = None
        #return yolov1_optim, yolov1_lr_scheduler

        return yolov1_optim
