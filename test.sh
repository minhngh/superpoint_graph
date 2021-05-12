python learning/train_custom.py --dataset sema3d --SEMA3D_PATH /home/kalinaxl/Documents/superpoint_graph/sema3d --db_test_name testred --db_train_name trainval \
--epochs 100 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --pc_attrib xyzrgbelpsv --odir "results/sema3d/trainval_best"
