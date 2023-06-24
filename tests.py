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
    print(f"\t Testing the shape of model's outputs.")
    model = YOLOV1(IMG_CHS,NUM_CLASSES,SPLIT_SIZE,NUM_BOXES)
    data = torch.rand(BATCH_SIZE,IMGS_CHS,IMG_H,IMG_W)
    outputs = model(data)
    outputs = outputs.view(BATCH_SIZE, SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*DATA_PER_BOX)

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

def run_tests():
    model_output_shape_test()
    yolov1_loss_test()
    test_voc_dataset_class()
    return

if __name__ == '__main__':
    run_tests()
