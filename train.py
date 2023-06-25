import torch
import torch.optim as optim
from datasets.classes            import VOCDataset
from torch.utils.data.dataloader import DataLoader
from training                    import YOLOV1Loss, Config
from models                      import YOLOV1
from tqdm                        import tqdm
from training.utils              import get_bboxes, load_tensorboard_writer, load_checkpoint, save_checkpoint, test_model, load_dataset
from training.metrics            import mean_average_precision

# Enables cuDNN (CUDA Deep Neural Network library) to find the best algorithm
# configuration for the specific input size and hardware that is being used.
torch.backends.cudnn.benchmark = True
seed = 123
torch.manual_seed(seed)

if __name__== "__main__":

    print('Training loop for YOLO v1.')

    # Load the config
    config = Config(verbose=True)

    device = None
    if config.general.cpu:
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for training.')

    # Load the dataset
    train_dataset = load_dataset(config.dataparams, split='train')
    test_dataset = load_dataset(config.dataparams, split='test')

    # Drop last = 1 as not doing it may result in unstable training
    train_dataloader = DataLoader(
        train_dataset,
        config.hyperparams.batch_size,
        num_workers=config.general.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    test_dataloader = DataLoader(
        test_dataset,
        config.hyperparams.batch_size,
        num_workers=config.general.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    # Load the model
    model = YOLOV1(
        in_chs=train_dataset[0][0].size()[0],
        num_classes=train_dataset.C,
        split_size=config.dataparams.img_split_size,
        num_boxes=config.dataparams.box_per_split).to(device)

    # Create the tensorboard writer
    writer, weigths_folder = load_tensorboard_writer(config)
    total_train_baches = int(len(train_dataloader))

    # Load the criterion and optimizer
    criterion = YOLOV1Loss(
        split_size=config.dataparams.img_split_size,
        num_classes=train_dataset.C,
        num_boxes=config.dataparams.box_per_split,
        lambda_coord=config.hyperparams.loss_lambda_coord,
        lambda_noobj=config.hyperparams.loss_lambda_nocoord)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.hyperparams.lr,
        weight_decay=config.hyperparams.weight_decay)

    # Load pretrained weigths for the model
    if config.general.pretrained_weights is not None:
        load_checkpoint(config.general.pretrained_weights, model, optimizer)

    print(f'The model will be tested every {config.general.test_model_epoch} epochs.')
    print(f'The mAP for the train and test datasets will be calculated every {config.general.test_model_epoch} epochs.')
    print(f"Check the training status with tensorboard. The tensorboard folder is runs/tensorboard/{weigths_folder.split('/')[-1]}\n\n")

    # Start of the training process
    step = 0
    best_test_mAP = 0.0
    for epoch in range(config.hyperparams.epochs):
        mean_train_epoch_loss = []
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            predictions = model(imgs)
            train_loss = criterion(predictions, labels)
            mean_train_epoch_loss.append(train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update progress bar
            writer.add_scalars(f'Loss/', {'Train': train_loss}, step)
            step += 1

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}/{config.hyperparams.epochs} ==> Batch: {batch_idx}/{len(train_dataloader)} ==> Mean loss: {sum(mean_train_epoch_loss)/len(mean_train_epoch_loss):.4f}', end='\r')

        if epoch % config.general.test_model_epoch == 0:
            with torch.no_grad():

                # Calculate train_mAP
                pred_boxes, labels_boxes = get_bboxes( train_dataloader, model, iou_threshold=config.hyperparams.iou_threshold, threshold=0.4, box_format=config.hyperparams.box_format)
                train_mAP = mean_average_precision(pred_boxes, labels_boxes, iou_threshold=config.hyperparams.iou_threshold, box_format=config.hyperparams.box_format)

                # Calculate test_mAP
                pred_boxes, labels_boxes = get_bboxes( test_dataloader, model, iou_threshold=config.hyperparams.iou_threshold, threshold=0.4)
                test_mAP = mean_average_precision(pred_boxes, labels_boxes, iou_threshold=config.hyperparams.iou_threshold, box_format=config.hyperparams.box_format)
                writer.add_scalars(f'mAP/', {'Train': train_mAP, 'Test': test_mAP}, step)
                print(f'************* Epoch {epoch}: Train mAP = {train_mAP:.4f} --- Test mAP = {test_mAP:.4f} --- Mean loss = {sum(mean_train_epoch_loss)/len(mean_train_epoch_loss):.4f}  *************')

                mean_test_loss = test_model(model, test_dataloader, criterion)
                writer.add_scalars(f'Loss/', {'Mean test': mean_test_loss}, step)

                if test_mAP > best_test_mAP:
                    best_test_mAP = test_mAP
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()}
                    save_checkpoint(checkpoint, f'{weigths_folder}/yolov1_{config.dataparams.dataset_name}_e{epoch}_mAPtest{test_mAP}.pt')
