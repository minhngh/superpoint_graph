CUDA_VISIBLE_DEVICES=0 python learning/train_custom.py --batch_size 1 --dataset sema3d --SEMA3D_PATH sema3d --db_test_name testred --db_train_name trainval \
        --epochs 700 --lr_steps '[300, 400, 500, 600, 650]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
        --nworkers 12 --pc_attrib xyzrgbelpsv --odir "results/sema3d/trainval_custom_best" > custom_log_sub_1.txt
