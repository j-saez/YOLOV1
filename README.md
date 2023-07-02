# YOLO V1

This is a YOLO v1 pytorch implementation. The main purpose of this repository was to have a better understanding of the YOLO algorithm and keep forging my pytorch skills.

## Model

In the [original paper](https://arxiv.org/abs/1506.02640), the authors use as a backbone an architecture that they called Darknet19. They said that the convolutional layers are pretrained on the ImageNet classification task at half the resolution of YOLO'S input (224x224) and then double the resolution for detection (448x448). This is YOLO v1 original architecture:

**ADD IMAGE**

Due to lack of hardware and time, in the results presented in this repository, Darknet19 was not trained on ImageNet. However, other architectures were used as backbones: ResNet50, ResNet34 and ResNet18. These other backbones were trained on ImageNet before performing detection.

## Loss function

The final layer of the model predicts both class probabilited and bounding box coordinates. The **bounding box width and height** is **normalized** by the image width and height so that they fall between 0 and 1. The **bounding box x and y coordinates** are **parametrized to be offsets** of a **parituclar grid cell** location so they are also bounded between 0 and 1. This means that the bounding boxes predicted by the model will be relative to the cell and not relative to the image.
The authors optimizsed sum-squared error in the output of our model as it is easy to optimize. However, this is not ideal as it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with calssification error which may not be ideal. Also, in every image many gird cells do not contain any object. This pushses the "confidence" scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability causeing training to diverge early on.
To remedy this, the loss from bounding box coordinate predictions was increased and the loss from confidence predictions for boxes that do not contain objects was decreased. This was done by adding two paramters (lambda_coord and lambda_nocoord).
Sum-squared erros also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this, the square root of the bounding box width and height was predited, instead of the width and height directly.
As well, YOLO predicts multiple bounding boxes per grid cell. At training time, it is just one bounding box predictor the respoinsible for each object. It is for that reason, that one predicto is assigned to be responsible for predicting an object based on which prediction has the highest current IoU with the ground truth. So, to sum up, the loss function takes into account only the bounding box with the highest IoU per each cell. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios or classes of object, improving overall recall.

The loss function used for the results published in the original paper was:

**ADD IMAGE**

Where: 1^{obj}_{i} denotes if object apperas in cell i and 1^{obj}_{ij} denotes that the jth bounding box predictor in cell i is responsible for that prediction.

Note that the loss function only penalizes classification error is an object is present in that grid cell. It also only penalizes bounding box coordinate eror if that predictor is respoinsible for the gound truth box (i.e. has the highest IoU of any predictor in that grid cell).

# Datasets

At the moment, it is only possible to train this algorithm with the VOC dataset. You can download it from (ADD LINK). Once you have downloaded, make sure it is placed inside the datasets/data directory.
It is not necessary to add it there. If you prefer having the dataset in any other path, add a symbolik link from that path to dataset/data:

```bash
ln -s /path/to/source/directory/VOC <yolov1 repo>/dataset/data/VOC
```

# Train

The train script allows to modify a few parameters. The default values are the ones that appears in the paper:

* Learning rate: --lr <value> (Default = 2e-5)
* Batch size: --batch-size <value> (Default = 16)
* Weight decay for the Adam optimizer: --weight-decay <value> (Default = 0.0)
* Dataset name: --dataset-name <value> (Default = voc) **At the moment voc is the only dataset allowed**
* Input image width: --input-img-h <value> (Default = 448)
* Input image height: --input-img-w <value> (Default = 448)

```bash
cd <root of the repo>
python train.py
```

# Inference

**TODO**

# Results

**TODO**
