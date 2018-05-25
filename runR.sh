#!/bin/bash

python prog2.py \
    -train_feat prog2_data/regression/multivariate/dataset$1.train_features.txt \
    -train_target prog2_data/regression/multivariate/multivariate_dataset$1.train_targets.txt \
    -dev_feat prog2_data/regression/multivariate/dataset$1.dev_features.txt \
    -dev_target prog2_data/regression/multivariate/multivariate_dataset$1.dev_targets.txt \
    -type R \
    -nunits 10 \
    -hidden_act tanh \
    -learnrate 0.001 \
    -optimizer grad \
    -init_range 1.0 \
    -epochs 100 \
    -nlayers 15 \
