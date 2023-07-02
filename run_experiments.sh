#!/bin/bash

# Arrays declaration
backbones=("resnet18" "resnet34" "resnet50" "darknet19")

# Start experiments
for ((i=0; i<${#backbones[@]}; i++)); do
    python train.py --backbone  "${backbones[t]}"\
                    --epochs  200\
                    --batch-size 8
done
