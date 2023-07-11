#!/bin/bash
#SBATCH -D /path/to/FL_bench
#SBATCH -J batch256
#SBATCH --partition=gtai
#SBATCH --nodelist=gpu05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16             
#SBATCH --mem=200GB                    # CPU RAM
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.


flight env activate conda
conda activate flare

RESULT_ROOT=/path/to/localscratch/JulyTasks
algorithms_dir="${PWD}/fl_bench/cifar10/job_configs"

# python3 fl_bench/utils/set_update_parameters.py \
# -job fl_bench/cifar10/job_configs/cifar10_fedopt \
# -dataset cifar10 \
# --alpha 0.6 \
# --num_rounds 50 \
# --lr 0.01 \
# --epochs 2 \
# --model_name resnet20
# nvflare simulator ${algorithms_dir}/cifar10_fedopt -w ${RESULT_ROOT}/resnet20 -n 10 -t 10 -gpu 0



# python3 fl_bench/utils/set_update_parameters.py \
# -job fl_bench/cifar10/job_configs/cifar10_fedavg \
# -dataset cifar10 \
# -learner adamw \
# --alpha 0.6 \
# --num_rounds 150 \
# --lr 0.001 \
# --epochs 4 \
# --model_name focalnet
# nvflare simulator ${algorithms_dir}/cifar10_fedavg -w ${RESULT_ROOT}/FocalNet -n 10 -gpu 0 -t 10

# python3 fl_bench/utils/set_update_parameters.py \
# -job fl_bench/cifar10/job_configs/cifar10_fedavg \
# -dataset cifar10 \
# -learner adamw \
# --alpha 0.6 \
# --num_rounds 150 \
# --lr 0.001 \
# --epochs 4 \
# --model_name simple_vit
# nvflare simulator ${algorithms_dir}/cifar10_fedavg -w ${RESULT_ROOT}/Simple_ViT -n 10 -gpu 0 -t 10

# python3 fl_bench/utils/set_update_parameters.py \
# -job fl_bench/cifar10/job_configs/cifar10_fedavg \
# -dataset cifar10 \
# --alpha 0.6 \
# --num_rounds 500 \
# --lr 0.01 \
# --epochs 2 \
# --model_name simple_vit_half
# nvflare simulator ${algorithms_dir}/cifar10_fedavg -w ${RESULT_ROOT}/Simple_ViT_half_noadamw -n 10 -gpu 0 -t 10

python3 fl_bench/utils/set_update_parameters.py \
-job fl_bench/cifar10/job_configs/cifar10_central \
-dataset cifar10 \
-learner savemore \
--alpha 0.6 \
-save_freq 2 \
--num_rounds 1 \
--lr 0.001 \
--epochs 200 \
--model_name swin
nvflare simulator ${algorithms_dir}/cifar10_central -w ${RESULT_ROOT}/Adamw_SWIN_central_lr_3_w_1_5 -n 1 -gpu 0

# python3 fl_bench/utils/set_update_parameters.py \
# -job fl_bench/cifar10/job_configs/cifar10_central \
# -dataset cifar10 \
# -learner savemore \
# --alpha 0.6 \
# -save_freq 2 \
# --num_rounds 1 \
# --lr 0.001 \
# --epochs 400 \
# --model_name swin
# nvflare simulator ${algorithms_dir}/cifar10_central -w ${RESULT_ROOT}/Adamw_SWIN_central_lr_3_w_1_5 -n 1 -gpu 0

python3 fl_bench/utils/set_update_parameters.py \
-job fl_bench/cifar10/job_configs/cifar10_scaffold \
-dataset cifar10 \
-learner scaffold \
--alpha 0.6 \
-save_freq 2 \
--num_rounds 250 \
--lr 0.01 \
--epochs 4 \
--model_name wideresnet_16_8
nvflare simulator ${algorithms_dir}/cifar10_scaffold -w ${RESULT_ROOT}/SCAFFOLD/wide_sanity_2 -n 10 -gpu 1 -t 10