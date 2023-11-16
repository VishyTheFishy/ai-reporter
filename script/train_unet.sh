#!/bin/bash
export CUDA_VISIBLE_DEVICES

set -e
# INPUT=$1 # PC_AI, BF
# OUTPUT=$2 # Nucleus, TUFM, CD29
# DATA_DIR="/home/andrewbai/data/cell_img2img/${OUTPUT}_MSC_20x_${INPUT}"
SRC_DIR=$1 # "${DATA_DIR}/input/train"
TGT_DIR=$2 # "${DATA_DIR}/output/train"
CKPT_FILE=$3 # "./checkpoints/baseline_${OUTPUT}_${INPUT}.ckpt"

BSIZE="2"
LR="1e-3"
EPOCHS="25"
WEIGHT_DECAY="1e-5"

python -m BNNBench.trainer.ensemble_trainer \
  --src-dir $SRC_DIR --tgt-dir $TGT_DIR \
  --dropout --batch-size $BSIZE --img-size 1024 \
  --init-lr $LR --total-epochs $EPOCHS \
  --weight-decay $WEIGHT_DECAY --ckpt-file $CKPT_FILE

