import torch
from models import YOLOV1
from training.loss import YOLOV1Loss
from datasets.classes.voc import IMGS_CHS, VOCDataset

IMG_H        = 448
IMG_W        = 448
IMG_CHS      = 3
BATCH_SIZE   = 50
SPLIT_SIZE   = 7
NUM_CLASSES  = 20
NUM_BOXES    = 2
DATA_PER_BOX = 5 # The 5 values are: (prob,x,y,w,h)


def test_voc_dataset_class():
    print(f"\t Testing the voc dataset class.")
    dataset = VOCDataset(
        data_split = 'train',
        img_split_size = SPLIT_SIZE,
        box_per_split = NUM_BOXES,
        model_in_w = IMG_W,
        model_in_h = IMG_H,)

    idx = torch.randint(low=0,high=len(dataset),size=(1,)).item()
    assert dataset[idx][0].shape == (IMG_CHS, IMG_H, IMG_W)
    assert dataset[idx][1].shape == (SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*5)

    print(f'\t\t Test passed.')
    return

def model_output_shape_test():
    inputs = torch.rand(BATCH_SIZE,IMG_CHS,IMG_H,IMG_W)
    for backbone in ['resnet18', 'resnet34', 'resnet50', 'darknet19']:
        model = YOLOV1(in_chs=3, num_classes=20,split_size=7,num_boxes=2,backbone_to_use=backbone)
        outputs = model(inputs)
        outputs = outputs.view(BATCH_SIZE, SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*DATA_PER_BOX)
        print(f'{backbone} outputs shape = {outputs.shape}')
        assert outputs.size() == (BATCH_SIZE, SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*DATA_PER_BOX)

    print(f'\t\t Test passed.')
    return

def yolov1_loss_test():
    print(f"\t Testing the loss function (allows backprop).")
    data = torch.rand(BATCH_SIZE,IMGS_CHS,IMG_H,IMG_W)
    labels = torch.rand((BATCH_SIZE, SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*DATA_PER_BOX), requires_grad=True)

    model = YOLOV1(in_chs=IMG_CHS,num_classes=20,split_size=7,num_boxes=2)
    predictions = model(data)
    predictions = predictions.view(BATCH_SIZE, SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*DATA_PER_BOX)

    criterion = YOLOV1Loss(
        SPLIT_SIZE,
        NUM_CLASSES,
        NUM_BOXES,
        lambda_coord=5,
        lambda_noobj=0.5)

    loss = criterion(predictions, labels)
    loss.backward()
    print(f'\t\t Test passed.')
    return

def mAP_test():
    from training.utils import get_bboxes
    from datasets.classes import VOCDataset
    from torch.utils.data import DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = VOCDataset(
        data_split = 'train',
        img_split_size = 7,
        box_per_split = 2,
        model_in_w = 448,
        model_in_h = 448)
    train_dataloader = DataLoader(train_dataset,batch_size=8,num_workers=2,pin_memory=True,shuffle=True,drop_last=True)
    model = YOLOV1(in_chs=3, num_classes=20, split_size=7, num_boxes=2, backbone_to_use='resnet18').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion=YOLOV1Loss(7,20,2,5,0.5)

    pred_boxes, labels_boxes = get_bboxes(train_dataloader, model, iou_threshold=0.5, threshold=0.4, box_format='midpoint', device=device)

    for i in range(50): 
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            predictions = model(imgs)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 5 ==0:
            pred_boxes, labels_boxes = get_bboxes(train_dataloader, model, iou_threshold=0.5, threshold=0.4, box_format='midpoint', device=device)
            print(f'pred_boxes shape = {torch.tensor(pred_boxes).shape}')
            """
            preds = [
                dict(
                    boxes = torch.tensor(pred_boxes[])
                )
            ]
            """

    return

def run_tests():
    """
    model_output_shape_test()
    yolov1_loss_test()
    test_voc_dataset_class()
    """
    mAP_test()
    return

if __name__ == '__main__':
    run_tests()
