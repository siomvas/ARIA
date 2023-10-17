#!/bin/bash
#SBATCH -D /path/to/FL_bench/fl_bench/medmnist
#SBATCH -J fed_1
#SBATCH --partition=gengpu
#SBATCH --nodelist=gpu04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=6             
#SBATCH --mem=40GB                    # CPU RAM
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.

flight env activate conda
conda activate flare

RESULT_ROOT=../../../localscratch/medmnist/
algorithms_dir="${PWD}/job_configs/hyperion_1"

dset="organamnist"
alpha=100
init="scratch"
agg=fedopt
seed=111
# COVERS WRN, VIT, SWIN, DENSENET, EFFICIENTNET, CONVNEXT

# INIT SCRATCH INIT SUPERVISED

# METHOD BLOCK START
prefix="${dset}_alpha=${alpha}_${init}_${agg}_"

#### MODEL BLOCK 1 START

model_name="wide_resnet_50_2"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 1 END

#### MODEL BLOCK 2 START

model_name="swin_t"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 2 END

#### MODEL BLOCK 3 START

model_name="vit_b_16"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 3 END

#### MODEL BLOCK 4 START

model_name="densenet_121"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 4 END

#### MODEL BLOCK 5 START

model_name="efficientnet_v2_s"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 5 END

#### MODEL BLOCK 6 START

model_name="convnext_small"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 6 END

# ## METHOD BLOCK END

# # METHOD BLOCK START
init="supervised"
prefix="${dset}_alpha=${alpha}_${init}_${agg}_"

#### MODEL BLOCK 1 START

model_name="wide_resnet_50_2_pretrained_pt"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 1 END

#### MODEL BLOCK 2 START

model_name="swin_t_pretrained_pt"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 2 END

#### MODEL BLOCK 3 START

model_name="vit_b_16_pretrained_pt"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 3 END

#### MODEL BLOCK 4 START

model_name="densenet_121_pretrained_pt"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 4 END

#### MODEL BLOCK 5 START

model_name="efficientnet_v2_s_pretrained_pt"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 5 END

#### MODEL BLOCK 6 START

model_name="convnext_small_pretrained_pt"
name="${prefix}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
--use_hyperion_dirs
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,0,0 \

#### MODEL BLOCK 6 END