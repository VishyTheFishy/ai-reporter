#!/bin/bash
export CUDA_VISIBLE_DEVICES

set -e

DATA_DIR=$1 # PC_AI, BF | Nucleus, TUFM, CD29
CKPT_FILE=$2
SAVE_DIR=$3
ADDITIONAL_FLAGS=$4 # "--load_from_pl --save_results_dir [DIR]
SPLIT="test"

SRC_DIR="$DATA_DIR/input/${SPLIT}"
TGT_DIR="$DATA_DIR/output/${SPLIT}"

echo "SRC_DIR: $SRC_DIR"
echo "TGT_DIR: $TGT_DIR"
echo "CKPT_FILE: $CKPT_FILE"

python eval_unet.py 
  --src-dir $SRC_DIR 
  --tgt-dir $TGT_DIR
  --img-size 1024 
  --ckpt-file $CKPT_FILE 
  --save_results_dir $SAVE_DIR
  $SAVE_DIR $ADDITIONAL_FLAGS

