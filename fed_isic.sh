RESULT_ROOT="/path/to/Results/FedOpt"
algo_dir="${PWD}/fl_bench/fed_isic/job_configs"

python fl_bench/utils/set_update_train_config.py \
-job fl_bench/fed_isic/job_configs/fed_isic_central \
-model_name efficientnet_scratch \
-n 5 \
-e 20 \
-lr 5e-4

nvflare simulator ${algorithms_dir}/fed_isic_central -w ${RESULT_ROOT}/FromScratch -n 1 -t 1 -gpu 1