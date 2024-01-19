DATASET=$1
TRIALS=$2
EPOCHS=$3

for trial in $(seq 0 $((TRIALS-1)))
do
mkdir log${DATASET}${trial}
for i in {1..16..1}
do 
  ./train_adda.sh /home/vishrutsgoyal/${DATASET}_MSC_20x_PC /home/vishrutsgoyal/${DATASET}_MSC_20x_BF log${DATASET}${trial}/layer${i}.ckpt ./baseline_${DATASET}_PC.ckpt "--max_epochs ${EPOCHS} --adaptation_layer ${i}"
done
done
