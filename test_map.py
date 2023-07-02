import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
preds = [
  dict(
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
    scores=torch.tensor([0.536]),
    labels=torch.tensor([0]),
  )
]
target = [
  dict(
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
    labels=torch.tensor([0]),
  )
]
metric = MeanAveragePrecision()
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())
