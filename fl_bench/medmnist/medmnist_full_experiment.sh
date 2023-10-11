conda activate flare

RESULT_ROOT=/path/to/localscratch/MedMNIST/
algorithms_dir="${PWD}/fl_bench/medmnist/job_configs"
name="direct_session"

# Central

# # Resnet
name="resnet_111"
python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0


python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

python3 ${PWD}/fl_bench/utils/set_update_train_config.py \
-job ${algorithms_dir}/medmnist_central \
-dataset medmnist \
--alpha 0.6 \
--num_rounds 20 \
--lr 0.03 \
--epochs 4 \
--model_name resnet20 \
--seed 111

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

