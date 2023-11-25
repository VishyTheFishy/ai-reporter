./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer1 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 1 --accumulate_grad_batches 8'

./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer3 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 3 --accumulate_grad_batches 8'


./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer5 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 5 --accumulate_grad_batches 8'


./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer7 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 7 --accumulate_grad_batches 8'


./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer9 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 9 --accumulate_grad_batches 8'


./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer11 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 11 --accumulate_grad_batches 8'


./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer13 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 13 --accumulate_grad_batches 8'


./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layer15 ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer 15 --accumulate_grad_batches 8'

./train_adda.sh /home/vishrutsgoyal/CD29_MSC_20x_BF /home/vishrutsgoyal/CD29_MSC_20x_PC logs_CD29_4/log_ADDA_BFtoPC_AI_layerNone ./baseline_CD29_BF.ckpt '--max_epochs 100 --adaptation_layer -1 --accumulate_grad_batches 8'

