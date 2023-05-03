#!/usr/bin/env bash

#module load cuda/11.3
#module load cudnn/7.6.5.32_cuda10.2 
#source activate graph


#python main.py --lr 0.012 --num_workers 5 --gpu 0 --batch 8192 --opn 'sub' --init_dim 50  --score_func 'dist' --embed_dim 50
python main.py --epoch 1500  --model_name ragat --score_func \
interacte --opn cross --gpu 0 --gcn_drop 0.4 --ifeat_drop 0.4 \
--ihid_drop 0.3 --batch 1024 --iker_sz 3 --attention True --head_num 2