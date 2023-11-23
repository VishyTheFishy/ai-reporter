
./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer1 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 1 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer3 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 3 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer5 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 5 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer7 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 7 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer9 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 9 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer11 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 11 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer13 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 13 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layer15 ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer 15 --accumulate_grad_batches 4'

./train_adda.sh /home/vishrutsgoyal/Nucleus_MSC_20x_BF /home/vishrutsgoyal/Nucleus_MSC_20x_PC_AI logs/log_ADDA_BFtoPC_AI_layerNone ./baseline_Nucleus_BF.ckpt '--max_epochs 10 --adaptation_layer -1 --accumulate_grad_batches 4'
