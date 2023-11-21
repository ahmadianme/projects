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

# normal
python src/xtec.py --train train,nominival --valid minival --llayers 9 --xlayers 5 --rlayers 5 --loadLXMERT snap/pretrained/model --batchSize 10 --optim bert --lr 5e-5 --epochs 4 --tqdm --multiGPU --numWorkers 1 --output snap/xtec/xtec_lxr955
# tiny
python src/xtec.py --train train,nominival --valid minival --llayers 9 --xlayers 5 --rlayers 5 --loadLXMERT snap/pretrained/model --batchSize 10 --optim bert --lr 5e-5 --epochs 4 --tqdm --multiGPU --numWorkers 1 --output snap/xtec/xtec_lxr955_tiny --tiny



# not multi gpu
# normal
python src/xtec.py --train train --valid minival --llayers 9 --xlayers 5 --rlayers 5 --loadLXMERT snap/pretrained/model --batchSize 10 --optim bert --lr 5e-5 --epochs 4 --tqdm --numWorkers 1 --output snap/xtec/xtec_lxr955
# tiny
python src/xtec.py --train train,nominival --valid minival --llayers 9 --xlayers 5 --rlayers 5 --loadLXMERT snap/pretrained/model --batchSize 10 --optim bert --lr 5e-5 --epochs 4 --tqdm --numWorkers 1 --output snap/xtec/xtec_lxr955_tiny --tiny



# nazari
python src/xtec.py --train train --valid minival --llayers 5 --xlayers 3 --rlayers 3 --batchSize 8 --optim bert --lr 5e-5 --epochs 100 --tqdm --output snap/xtec/nazari
