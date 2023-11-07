#!/bin/bash
#SBATCH -D /path/to/FL_bench/fl_bench/extra
#SBATCH -J ssl_comparison
#SBATCH --partition=gengpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8             
#SBATCH --mem=40GB                    # CPU RAM
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.

flight env activate conda
conda activate flare
RESULT_ROOT=/path/to/extra/
seed=42

# FEDISIC SECTION
algorithms_dir="/path/to/FL_bench/fl_bench/fed_isic/job_configs"

# # vit moco fedisic
# model_name=vit_b_16_pretrained_mocov3

# name="fed_isic_${model_name}_${seed}_ssl_comparison"

# python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated_extra \
# -method fedavg \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed ${seed} \
# --amp \
# --use_hyperion_dirs \
# --opt_args '{"class": "Adam"}'


# nvflare simulator ${algorithms_dir}/fed_isic_federated_extra -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

# # vit dino fedisic
# model_name=vit_b_16_pretrained_dino

# name="fed_isic_${model_name}_${seed}_ssl_comparison"

# python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated_extra \
# -method fedavg \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed ${seed} \
# --amp \
# --use_hyperion_dirs \
# --opt_args '{"class": "Adam"}'
# nvflare simulator ${algorithms_dir}/fed_isic_federated_extra -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

# r50 dino fedisic
model_name=resnet_50_pretrained_dino

name="fed_isic_${model_name}_${seed}_ssl_comparison"

python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated_extra \
-method fedavg \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated_extra -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

# r50 moco fedisic
model_name=resnet_50_pretrained_mocov3

name="fed_isic_${model_name}_${seed}_ssl_comparison"

python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated_extra \
-method fedavg \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated_extra -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

# MEDMNIST SECTION

algorithms_dir="/path/to/FL_bench/fl_bench/medmnist/job_configs"

# vit moco medmnist
model_name=vit_b_16_pretrained_mocov3
name="medmnist_${model_name}_${seed}_ssl_comparison"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_federated_extra \
-method fedavg \
-alpha 0.2 \
-num_rounds 25 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated_extra -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,1,1

# vit dino medmnist
model_name=vit_b_16_pretrained_dino
name="medmnist_${model_name}_${seed}_ssl_comparison"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_federated_extra \
-method fedavg \
-alpha 0.2 \
-num_rounds 25 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated_extra -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,1,1


# resnet50 moco medmnist
model_name=resnet_50_pretrained_mocov3
name="medmnist_${model_name}_${seed}_ssl_comparison"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_federated_extra \
-method fedavg \
-alpha 0.2 \
-num_rounds 25 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated_extra -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,1,1

# resnet50 dino medmnist
model_name=resnet_50_pretrained_dino
name="medmnist_${model_name}_${seed}_ssl_comparison"
python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
-job ${algorithms_dir}/medmnist_federated_extra \
-method fedavg \
-alpha 0.2 \
-num_rounds 25 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--use_hyperion_dirs \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated_extra -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,1,1