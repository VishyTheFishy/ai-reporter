DATASET="CD29"
TRIALS=2
EPOCHS=20

for trial in $(seq 0 $((TRIALS-1)))
do
mkdir log_transfer${DATASET}${trial}
for i in {0..15..1}
do 
  ./train_transfer.sh /home/vishrutsgoyal/${DATASET}_MSC_20x_PC /home/vishrutsgoyal/${DATASET}_MSC_20x_BF log_transfer${DATASET}${trial}/layer${i}.ckpt ./baseline_${DATASET}_PC.ckpt "--max_epochs ${EPOCHS} --adaptation_layer ${i}"
done
done
