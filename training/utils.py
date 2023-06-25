import os
import torch
import datasets.classes as dataset_classes
from training.metrics        import non_max_suppression
from torch.utils.tensorboard import SummaryWriter
from training                import YOLOV1Loss, Dataparams

PROB_IDX = 1
AVAILABLE_DATASETS = ['voc']

def convert_cellboxes(predictions, S=7):
    """
    TODO

    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def get_bboxes(loader: torch.utils.data.DataLoader, model: torch.nn.Module, iou_threshold: float, threshold: float, box_format: str="midpoint", device: str="cuda"):
    """
    Gets all the prediction and label boxes for the data passed in the loader parameter RELATIVE to the IMAGE.
    Inputs: 
        >> loader: (DataLoader) Contains the images and labels.
        >> model: (nn.Module) Object detector
        >> iou_threshold: (float) Threshold where the predicted boxes are correct.
        >> threshold: (float) Threshold to remove predicted bboxes (independent of IoU).
        >> box_format: (str) "midpoint" [x,y,w,h] or "corners" [x1,y1,x2,y1].
        >> device: (str) Device where the data and the model will run.
    Outputs:
        >> all_pred_boxes: (list) Model predictions boxes. TODO: is it a list of lists?
        >> all_true_boxes: (list) Label boxes.             TODO: is it a list of lists?
    """
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[PROB_IDX] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cellboxes_to_boxes(out, S: int=7):
    """
    TODO
    Convert boxes relatives to a cell to boxes relatives to the image.
    Inputs:
        >> out: ()
        >> S: ()
    Outputs:
        >> all_bboxes: (list) Boxes relatives to the image.
    """
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state: dict, filename: str):
    """
    Saves the state of the model in the especified location.
    Inputs:
        state: (dict) Contains the state of the model in "state_dict" and the state of the optimizer in "optimizer".
        filename: (str) String contining the path and name of the file where the state will be stored.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("\t => Saved.")
    return


def load_checkpoint(weight_filename, model, optimizer):
    """
    Loads the checkpoint from the especified file and assigns the state of the model and the optimizer back.
    Inputs:
        >> weight_filename: (str) String containing the name and the path to the state file.
        >> model: (torch.nn.Module) Model for which the weights will be loaded and assigned back.
        >> optimizer: (torch.nn.optim) Optimizer used for the training process.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(weight_filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def load_tensorboard_writer(config):
    """
    Loads the tensorboard writer object.
    Inputs:
        >> config: configuration for the training (check <root repo>/training/config.py).
    Outputs:
        >> writer: tensorboard.SummaryWriter to save the training data.
        >> model_checkpoints_dir: (str) containing the path where the weights of the models will be saved.
    """
    tensorboard_dir = os.getcwd()+'/runs/tensorboard/'
    checkpoints_dir = os.getcwd()+'/runs/checkpoints/'
    training_dir_name = f'YOLO_{config.dataparams.dataset_name}_bs{config.hyperparams.batch_size}_lr{config.hyperparams.lr}_e{config.hyperparams.epochs}'
    model_logs_dir = os.getcwd() + '/runs/tensorboard/' + training_dir_name
    model_checkpoints_dir = os.getcwd() + '/runs/checkpoints/' + training_dir_name

    create_runs_dirs(tensorboard_dir, checkpoints_dir, model_checkpoints_dir, model_logs_dir)
    writer = SummaryWriter(log_dir=model_logs_dir)
    return writer, model_checkpoints_dir

def create_runs_dirs(tensorboard_dir, weights_dir, model_weights_dir, model_logs_dir) -> None:
    """
    Creates the directories to save all the data in tensorboards and the models' weights.
    Inputs:
        >> tensorboard_dir
        >> weights_dir
        >> model_weights_dir
        >> model_logs_dir
    Outputs: None
    """
    if not os.path.isdir(os.getcwd()+'/runs'):
        os.mkdir(os.getcwd()+'/runs')
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    if not os.path.isdir(model_logs_dir):
        os.mkdir(model_logs_dir)
    if not os.path.isdir(model_weights_dir):
        os.mkdir(model_weights_dir)
    return

def test_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: YOLOV1Loss):
    """
    Tests the model and returns the average loss over the test data.
    Inputs:
        >> model: (torch.nn.Module) Model to be tested.
        >> dataloader: (torch.utils.data.DataLoader) DataLoader containing the test data.
        >> criterion: (torch.nn.Module) Criterion for calculating the test loss.
    Outputs:
        >> test_avg_loss: (float) Average test loss.
    """
    loss_list = []
    device = model.device_param.device
    with torch.no_grad():
        model.eval()
        for imgs, labels in dataloader: 
            imgs = imgs.to(device)
            labels = labels.to(device)

            predictions = model(imgs)
            loss = criterion(predictions, labels)
            loss_list.append(loss)

    model.train()
    test_avg_loss = sum(loss_list) / len(loss_list)
    return test_avg_loss

def load_dataset(dataparams: Dataparams, split: str):
    """
    Loads the desired dataset with the parameters especified in the dataparams object.
    Inputs:
        >> dataparams: (training.config.Dataparams) Contains all the required params to load the dataset.
        >> split: (str) Split to be loaded (train or test).
    Outputs:
        >> dataset: (torch.utils.data.Dataset) Loaded dataset object.
    """
    if dataparams.dataset_name == 'voc':
        dataset = dataset_classes.VOCDataset(
            data_split = split,
            img_split_size = dataparams.img_split_size,
            box_per_split = dataparams.box_per_split,
            model_in_w = dataparams.model_in_h,
            model_in_h = dataparams.model_in_w)
    else:
        raise ValueError(f'{dataparams.dataset_name} is not a valid dataset name. The available ones are {AVAILABLE_DATASETS}')

    return dataset
