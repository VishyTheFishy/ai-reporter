#/bin/bash
export CUDA_VISIBLE_DEVICES
set -e

DIR_A=$1 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_A}"
DIR_B=$2 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_B}"
LOG_DIR=$3 # "./adda_logs/${SRC_B}_${TGT}"
UNET_PATH=$4 # "./checkpoints/baseline_${TGT}_${SRC_A}.ckpt"
KWARGS=$5 # --max_epochs 200 --save_every_n_epochs 20, --max_B_size 20, --name ntrn-100

mkdir -p $LOG_DIR

python train_adda.py --accelerator 'gpu' --auto_select_gpus 'True' --devices 1 \
  --data_dir_A $DIR_A --data_dir_B $DIR_B --pretrained_unet_path $UNET_PATH \
  --save_dir $LOG_DIR $KWARGS
