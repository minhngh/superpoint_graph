#!/bin/bash
mkdir gated_gcn_logs
for c in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do  
   CUDA_VISIBLE_DEVICES=0 python learning/train_custom.py --batch_size 1 --dataset sema3d --SEMA3D_PATH sema3d --db_test_name testred --db_train_name train \
        --epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
        --nworkers 12 --pc_attrib xyzrgbelpsv --k-fold ${c} --odir "results/sema3d/trainval_custom_best" > "gated_gcn_logs/custom_log_k_${c}.txt"
done


