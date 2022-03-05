## compared paper model 
# nohup ./run_arcf.sh >../experiment/arcf_1000_n16b64p64.txt &
# killed ps -ef |grep mainpppy |awk '{print $2}' |xargs kill -9

# no qm
# arcnn
# arcnn training on cufed dataset
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template arcnn --quality [5] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save arcnn_q5_100_b9p80_cufed --save_results --visdom --n_input 1

# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer2 --template arcnn --quality [10] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 3 --patch_size 80 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save arcnn_q10_100_b9p80_cufed --save_results --visdom --n_input 1

# dncnn
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template dncnn --quality [5] \
# --data_train CUFED_SINGLE --batch_size=9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save dncnn_q5_100_b9p80_cufed --save_results --visdom --n_input 1

# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template dncnn --quality [10] \
# --data_train CUFED_SINGLE --batch_size=9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save dncnn_q10_100_b9p80_cufed --save_results --visdom --n_input 1


# memnet
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template memnet --quality [5] \
# --data_train CUFED_SINGLE --batch_size=9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save memnet_q5_100_b9p80_cufed --save_results --visdom --n_input 1

# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template memnet --quality [10] \
# --data_train CUFED_SINGLE --batch_size=9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save memnet_q10_100_b9p80_cufed --save_results --visdom --n_input 1

# color
# rnan 
# q10
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template rnan --quality [10] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --load rnan_q10_100_b9p160_cufed --resume -1  --n_input 1 --chop --resume -1 --save_results \
# --visdom --save_models

# q5
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer2 --template rnan --quality [5] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save rnan_q5_100_b9p80_cufed --n_input 1 --chop --save_results \
# --visdom

# qmarg
# qmarg training on cufed dataset
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer2 --template qmarg --quality [5] --qm --n_resblocks 64 \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --res_scale 0.1 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save qmarg_q5_100_n64b9p80_cufed --save_results --visdom --n_input 1

# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer2 --template qmarg --quality [10] --qm --n_resblocks 64 \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --res_scale 0.1 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save qmarg_q10_100_n64b9p80_cufed --save_results --visdom --n_input 1


# rdn
# q10
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template rdn --quality [10] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --save rdn_q10_100_b9p80_cufed --n_input 1 --save_results \
# --save_models

# q5
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template rdn --quality [5] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --save rdn_q5_100_b9p80_cufed --n_input 1 --save_results --visdom \
# --save_models

# drunet
# q5
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template drunet --quality [5] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 --qv \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --save drunet_q5_100_b9p80_cufed --n_input 1 --save_results --visdom \
# --save_models

# q10
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template drunet --quality [10] \
# --data_train CUFED_SINGLE --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 --qv \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --save drunet_q10_100_b9p80_cufed --n_input 1 --save_results --visdom \
# --save_models

# mwcnn
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer2 --template mwcnn --quality [5] \
# --data_train CUFED_SINGLE --batch_size=9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [5] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save mwcnn_q5_100_b9p80_cufed --save_results --n_input 1

# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer2 --template mwcnn --quality [10] \
# --data_train CUFED_SINGLE --batch_size=9 --n_threads 9 --patch_size 80 \
# --data_test CUFED --test_quality [10] --loss 1*L1 \
# --epochs 90 --print_every 600 --test_every 100 \
# --save mwcnn_q10_100_b9p80_cufed --save_results --n_input 1