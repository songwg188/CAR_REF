## compared paper model 
# nohup ./run_arcf.sh >../experiment/arcf_1000_n16b64p64.txt &
# killed 
# ps -ef |grep mainpppy |awk '{print $2}' |xargs kill -9

# color
# q5
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test CUFED_REF --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test Sun80_REF --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test BSDS500_REF_DIV_q5 --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test LIVE1_REF_DIV_q5 --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test LIVE1_REF_WEB --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test Set5_REF --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test BSDS20_REF --res_scale 0.1 --test_quality [5] --load rarn_rs01_q5_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
# q10
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test CUFED_REF --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test Sun80_REF --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test BSDS500_REF_DIV_q10 --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test LIVE1_REF_DIV_q10 --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test Set5_REF --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test BSDS20_REF --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn --n_colors 3 --n_resblocks 32 --data_test B1_REF --res_scale 0.1 --test_quality [10] --load rarn_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only

# drunet
# q5
python main.py --template drunet --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --qv --load drunet_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --qv --load drunet_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --qv --load drunet_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --qv --load drunet_q5_100_b9p80_cufed --save_results --resume 90 --test_only
#q10
python main.py --template drunet --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --qv --load drunet_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --qv --load drunet_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --qv --load drunet_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --qv --load drunet_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template drunet --n_colors 3 --data_test B1 --test_quality [10] --n_input 1 --qv --load drunet_q10_100_b9p80_cufed --save_results --resume 90 --test_only

# rdn
# q5
python main.py --template rdn --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --load rdn_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --load rdn_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test BSDS500 --test_quality [5] --n_input 1 --load rdn_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test LIVE1 --test_quality [5] --n_input 1 --load rdn_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --load rdn_q5_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --load rdn_q5_100_b9p80_cufed --save_results --resume 90 --test_only
#q10
python main.py --template rdn --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test BSDS500 --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test LIVE1 --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only
python main.py --template rdn --n_colors 3 --data_test B1 --test_quality [10] --n_input 1 --load rdn_q10_100_b9p80_cufed --save_results --resume 90 --test_only

# qmarg
# q5
python main.py --template qmarg --n_colors 3 --data_test CUFED --test_quality [5] --qm --n_resblocks 64 --n_input 1 --load qmarg_q5_100_n64b9p80_cufed --save_results --resume -1 --test_only
python main.py --template qmarg --n_colors 3 --data_test Sun80 --test_quality [5] --qm --n_resblocks 64 --n_input 1 --load qmarg_q5_100_n64b9p80_cufed --save_results --resume -1 --test_only
python main.py --template qmarg --n_colors 3 --data_test Set5 --test_quality [5] --qm --n_resblocks 64 --n_input 1 --load qmarg_q5_100_n64b9p80_cufed --save_results --resume -1 --test_only
python main.py --template qmarg --n_colors 3 --data_test BSDS20 --test_quality [5] --qm --n_resblocks 64 --n_input 1 --load qmarg_q5_100_n64b9p80_cufed --save_results --resume -1 --test_only
#q10
python main.py --template qmarg --n_colors 3 --data_test CUFED --test_quality [10] --qm --n_resblocks 64 --n_input 1 --load qmarg_q10_100_n64b9p80_cufed --save_results --resume -1 --test_only
python main.py --template qmarg --n_colors 3 --data_test Sun80 --test_quality [10] --qm --n_resblocks 64 --n_input 1 --load qmarg_q10_100_n64b9p80_cufed --save_results --resume -1 --test_only
python main.py --template qmarg --n_colors 3 --data_test Set5 --test_quality [10] --qm --n_resblocks 64 --n_input 1 --load qmarg_q10_100_n64b9p80_cufed --save_results --resume -1 --test_only
python main.py --template qmarg --n_colors 3 --data_test BSDS20 --test_quality [10] --qm --n_resblocks 64 --n_input 1 --load qmarg_q10_100_n64b9p80_cufed --save_results --resume -1 --test_only

# rnan
# q5
python main.py --template rnan --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --load rnan_q5_100_b9p80_cufed --save_results --chop --resume -1 --test_only
python main.py --template rnan --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --load rnan_q5_100_b9p80_cufed --save_results --chop --resume -1 --test_only
python main.py --template rnan --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --load rnan_q5_100_b9p80_cufed --save_results --chop --resume -1 --test_only
python main.py --template rnan --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --load rnan_q5_100_b9p80_cufed --save_results --chop --resume -1 --test_only
#q10
python main.py --template rnan --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --load rnan_q10_100_b9p80_cufed --save_results --chop --resume 90 --test_only
python main.py --template rnan --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --load rnan_q10_100_b9p80_cufed --save_results --chop --resume 90 --test_only
python main.py --template rnan --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --load rnan_q10_100_b9p80_cufed --save_results --chop --resume 90 --test_only
python main.py --template rnan --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --load rnan_q10_100_b9p80_cufed --save_results --chop --resume 90 --test_only

# memnet
# q5
python main.py --template memnet --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --load memnet_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template memnet --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --load memnet_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template memnet --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --load memnet_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template memnet --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --load memnet_q5_100_b9p80_cufed --save_results --resume -1 --test_only
#q10
python main.py --template memnet --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --load memnet_q10_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template memnet --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --load memnet_q10_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template memnet --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --load memnet_q10_100_b9p80_cufed --save_results  --resume -1 --test_only
python main.py --template memnet --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --load memnet_q10_100_b9p80_cufed --save_results  --resume -1 --test_only

# dncnn
# q5
python main.py --template dncnn --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --load dncnn_q5_100_b9p80_cufed --save_results --resume -2 --test_only
python main.py --template dncnn --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --load dncnn_q5_100_b9p80_cufed --save_results --resume -2 --test_only
python main.py --template dncnn --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --load dncnn_q5_100_b9p80_cufed --save_results --resume -2 --test_only
python main.py --template dncnn --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --load dncnn_q5_100_b9p80_cufed --save_results --resume -2 --test_only
#q10
python main.py --template dncnn --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --load dncnn_q10_100_b9p80_cufed --save_results --resume -2 --test_only
python main.py --template dncnn --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --load dncnn_q10_100_b9p80_cufed --save_results --resume -2 --test_only
python main.py --template dncnn --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --load dncnn_q10_100_b9p80_cufed --save_results  --resume -2 --test_only
python main.py --template dncnn --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --load dncnn_q10_100_b9p80_cufed --save_results  --resume -2 --test_only

# arcnn
# q5
python main.py --template arcnn --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --load arcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template arcnn --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --load arcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template arcnn --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --load arcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template arcnn --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --load arcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
#q10
python main.py --template arcnn --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --load arcnn_q10_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template arcnn --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --load arcnn_q10_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template arcnn --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --load arcnn_q10_100_b9p80_cufed --save_results  --resume -1 --test_only
python main.py --template arcnn --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --load arcnn_q10_100_b9p80_cufed --save_results  --resume -1 --test_only

# q10
python main.py --template rarn_nfsp --n_colors 3 --n_resblocks 32 --data_test CUFED_REF --res_scale 0.1 --test_quality [10] --load rarn_nfsp_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn_nfsp --n_colors 3 --n_resblocks 32 --data_test Sun80_REF --res_scale 0.1 --test_quality [10] --load rarn_nfsp_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn_nfsp --n_colors 3 --n_resblocks 32 --data_test BSDS500_REF_DIV_q10 --res_scale 0.1 --test_quality [10] --load rarn_nfsp_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn_nfsp --n_colors 3 --n_resblocks 32 --data_test LIVE1_REF_DIV_q10 --res_scale 0.1 --test_quality [10] --load rarn_nfsp_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn_nfsp --n_colors 3 --n_resblocks 32 --data_test Set5_REF --res_scale 0.1 --test_quality [10] --load rarn_nfsp_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only
python main.py --template rarn_nfsp --n_colors 3 --n_resblocks 32 --data_test BSDS20_REF --res_scale 0.1 --test_quality [10] --load rarn_nfsp_rs01_q10_100_n128b9p80_cufed --save_results --n_input 2 --resume 90 --test_only


python main.py --template arcf22 --n_colors 3 --n_resblocks 32 --data_test BSDS20_REF --res_scale 0.1 --test_quality [10] --load arcf22_rs01_q10_100_n192b9p80 --save_results --n_input 2 --resume 98 --test_only

# mwcnn
# q5
python main.py --template mwcnn --n_colors 3 --data_test CUFED --test_quality [5] --n_input 1 --load mwcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template mwcnn --n_colors 3 --data_test Sun80 --test_quality [5] --n_input 1 --load mwcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template mwcnn --n_colors 3 --data_test Set5 --test_quality [5] --n_input 1 --load mwcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template mwcnn --n_colors 3 --data_test BSDS20 --test_quality [5] --n_input 1 --load mwcnn_q5_100_b9p80_cufed --save_results --resume -1 --test_only
#q10
python main.py --template mwcnn --n_colors 3 --data_test CUFED --test_quality [10] --n_input 1 --load mwcnn_q10_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template mwcnn --n_colors 3 --data_test Sun80 --test_quality [10] --n_input 1 --load mwcnn_q10_100_b9p80_cufed --save_results --resume -1 --test_only
python main.py --template mwcnn --n_colors 3 --data_test Set5 --test_quality [10] --n_input 1 --load mwcnn_q10_100_b9p80_cufed --save_results  --resume -1 --test_only
python main.py --template mwcnn --n_colors 3 --data_test BSDS20 --test_quality [10] --n_input 1 --load mwcnn_q10_100_b9p80_cufed --save_results  --resume -1 --test_only