#!/bin/bash
for c in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do  
    CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset sema3d --SEMA3D_PATH sema3d --db_test_name testred --db_train_name train \
    --epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
    --nworkers 2 --pc_attrib xyzrgbelpsv --odir "results/sema3d/trainval_best" --k-fold ${c} > "pp_log_k_${c}.txt"

done


