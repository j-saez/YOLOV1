#!/bin/bash

# Arrays declaration
backbones=("resnet18" "resnet34" "resnet50" "darknet19")

# Start experiments
for ((i=0; i<${#backbones[@]}; i++)); do
    python train.py --backbine  "${backbones[t]}"\
                    --epochs  200\
                    --batch_size 8
done
