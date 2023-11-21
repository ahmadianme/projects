#!/usr/bin/env bash
# The name of this experiment.
name=$2

# Save logs and models under snap/xtec; make backup.
output=snap/xtec/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/xtec.py \
    --train train \
    --valid '' \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 192 --optim bert --lr 5e-5 --epochs 100 \
    --numWorkers 1 \
    --maxObjectClassLength 15 \
    --tqdm --output $output ${@:3}


#--loadLXMERT snap/pretrained/model \
