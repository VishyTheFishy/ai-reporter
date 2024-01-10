DATASET=$1
TRIALS=$2
EPOCHS=$3

for trial in {0..${TRIALS}..1}
do
mkdir log${DATASET}${trial}
for i in {0..16..1}
do 
  ./train_adda.sh /home/vishrutsgoyal/${DATASET}_MSC_20x_BF /home/vishrutsgoyal/${DATASET}_MSC_20x_PC log${DATASET}${trial}/layer${i}.ckpt ./baseline_${DATASET}_BF.ckpt "--max_epochs ${EPOCHS} --adaptation_layer ${i} --accumulate_grad_batches 4"
done
done
