#!/bin/bash
#SBATCH -D /path/to/FL_bench/fl_bench/extra
#SBATCH -J organ
#SBATCH --partition=gengpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6             
#SBATCH --mem=20GB                    # CPU RAM
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.


source /opt/apps/flight/env/conda+default/etc/profile.d/conda.sh
conda activate flare
RESULT_ROOT=/path/to/sharedscratch/extra/


# MEDMNIST SECTION

algorithms_dir="/path/to/FL_bench/fl_bench/medmnist/job_configs/hyperion_1"

model_name=vit_b_16_ssl
seed=222
name="medmnist_${model_name}_${seed}"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_central \
-method fedavg \
-alpha -2 \
-num_rounds 2 \
-lr 0.01 \
--batch_size 64 \
-local_steps 10000 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'
nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0


model_name=vit_b_16_ssl
seed=111
name="medmnist_${model_name}_${seed}"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_central \
-method fedavg \
-alpha -2 \
-num_rounds 2 \
-lr 0.01 \
--batch_size 64 \
-local_steps 10000 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'
nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0


model_name=vit_b_16_supervised
seed=111
name="medmnist_${model_name}_${seed}"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_central \
-method fedavg \
-alpha -2 \
-num_rounds 2 \
-lr 0.01 \
--batch_size 64 \
-local_steps 10000 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'
nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0

model_name=vit_b_16_scratch
seed=111
name="medmnist_${model_name}_${seed}"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_central \
-method fedavg \
-alpha -2 \
-num_rounds 2 \
-lr 0.01 \
--batch_size 64 \
-local_steps 10000 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'
nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0