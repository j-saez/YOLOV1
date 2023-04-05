import torch
from models import YOLOV1
from training.loss import YOLOV1Loss
from datasets.classes.voc import VOCDataset

model = YOLOV1(in_chs=3,num_classes=20,split_size=7,num_boxes=2)
data = torch.rand(2,3,448,448)
outputs = model(data)
print(outputs.size())
print(outputs.view(2, 7, 7, 30).size())
print(outputs.size())

predictions = torch.rand((3,7,7,30),requires_grad=True)
labels = torch.rand((3,7,7,25),requires_grad=True)

print("javi's loss")
criterion = YOLOV1Loss(split_size=7,num_classes=20, num_boxes=2, lambda_coord=5, lambda_noobj=0.5)
loss = criterion(predictions, labels)
print(f'javi loss = {loss}')
loss.backward()

# Dataset test
dataset = VOCDataset(
    data_split = 'train',
    img_split_size = 7,
    box_per_split = 2,
    model_in_w = 448,
    model_in_h = 448,)
