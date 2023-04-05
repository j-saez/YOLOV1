# YOLO V1

This is a YOLO v1 pytorch implementation. The main purpose of this repository was to have a better understanding of the YOLO algorithm and keep forging my pytorch skills.

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
