import torch
import torch.optim as optim
from datasets.classes            import VOCDataset
from torch.utils.data.dataloader import DataLoader
from training                    import YOLOV1Loss, Config
from models                      import YOLOV1

seed = 10
torch.manual_seed(10)

if __name__== "__main__":
    print('Training loop for YOLO v1.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for training.')

    # Load the config
    config = Config(verbose=True)

    # Load the dataset
    train_dataset = VOCDataset(
        data_split = 'train',
        img_split_size = 7,
        box_per_split = 2,
        model_in_w = 448,
        model_in_h = 448,)

    test_dataset = VOCDataset(
        data_split = 'test',
        img_split_size = 7,
        box_per_split = 2,
        model_in_w = 448,
        model_in_h = 448,)

    train_dataloader = DataLoader(train_dataset, config.hyperparams.batch_size, shuffle=True, drop_last=True)

    # Load the model
    in_chs = train_dataset[0][0].size()[0]
    model = YOLOV1(in_chs, num_classes=train_dataset.C,split_size=7, num_boxes=2).to(device)

    # Load the criterion and optimizer
    criterion = YOLOV1Loss(split_size=7,num_classes=train_dataset.C, num_boxes=2, lambda_coord=5, lambda_noobj=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config.hyperparams.lr, weight_decay=config.hyperparams.weight_decay)

    # Start of the training process
    mean_loss = []
    print('Start of the training process')
    for epoch in range(config.hyperparams.epochs):
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            predictions = model(imgs)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_iou = calculate_iou(predictions[...,21:25])
                print(f'Epoch {epoch}/{config.hyperparams.epochs} -- batch_idx {batch_idx}/{len(train_dataloader)} -- Loss: {loss.mean().item()} -- IoU: {TODO}')

        if epoch % 5 == 0:
            with torch.no_grad():
                test_model()
