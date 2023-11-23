
./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer13 ./baseline_Nucleus_BF.ckpt '--max_epochs 15 --adaptation_layer 13 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer15 ./baseline_Nucleus_BF.ckpt '--max_epochs 15 --adaptation_layer 15 --accumulate_grad_batches 4'



