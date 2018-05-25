#!/bin/bash
epochs=$1
lr=$2
nunits=$3
hid_act=$4
opt=$5
init_range=$6
C=$7
nlayers=$8
mb=$9
python prog2.py \
    -train_feat prog2_data/classification/dataset$1.train_features.txt \
    -train_target prog2_data/classification/dataset$1.train_targets.txt \
    -dev_feat prog2_data/classification/dataset$1.dev_features.txt \
    -dev_target prog2_data/classification/dataset$1.dev_targets.txt \
    -nunits 64 \
    -num_classes $2 \
    -hidden_act sig \
    -type C \
    -learnrate 0.01 \
    -optimizer adam \
    -epochs 100 \
    -nlayers 10 \
    -init_range 0.025 \
    -mb 16 #-v
