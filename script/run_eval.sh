DATASET=$1
TRIALS=$2

for trial in $(seq 0 $((TRIALS-1)))
do
mkdir results${DATASET}${trial}
for i in {1..16..1}
do 
  ./eval_unet.sh /home/vishrutsgoyal/${DATASET}_MSC_20x_BF log${DATASET}${trial}/layer${i}.ckpt/version_0/checkpoints/last.ckpt  results${DATASET}${trial}/layer${i}
done
done
