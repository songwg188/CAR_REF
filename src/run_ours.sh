## my arcf 
# nohup ./run_arcf.sh >../experiment/arcf_1000_n16b64p64.txt &
# killed ps -ef |grep mainpppy |awk '{print $2}' |xargs kill -9

# color image
# q=10

# rarn 1+1/2+1/4+1/4+1/2+1 n_res 32  res_scale 0.1
# q10
# Computational complexity:       60.17 GMac
# Number of parameters:           14.64 M  
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer2 --template rarn --quality [10] --n_resblocks 32 \
# --data_train CUFED --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 --res_scale 0.1 \
# --data_test CUFED_REF --test_quality [10] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --save rarn_rs01_q10_100_n128b9p80_cufed --save_results --visdom --n_input 2 --save_models

# q5
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer2 --template rarn --quality [5] --n_resblocks 32 \
# --data_train CUFED --batch_size 9 --n_threads 9 --patch_size 80 --n_colors 3 --res_scale 0.1 \
# --data_test CUFED_REF --test_quality [5] --loss 1*L1 \
# --epochs 100 --print_every 600 --test_every 100 \
# --save rarn_rs01_q5_100_n128b9p80_cufed --save_results --visdom --n_input 2 --save_models
