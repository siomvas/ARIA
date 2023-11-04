conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

model_name=resnet_50_pretrained_timm
seed=2023

name="fed_isic_central_supervised_${model_name}_seed_${seed}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_central \
-method fedavg \
-num_rounds 2 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 2910 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'


nvflare simulator ${algorithms_dir}/fed_isic_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 1

name="fed_isic_supervised_fedavg_${model_name}_${seed}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method fedavg \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'


nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 1,1,1,1,1,1

# name="fed_isic_supervised_fedopt_${model_name}_${seed}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated \
# -method fedopt \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed ${seed} \
# --amp \
# --opt_args '{"class": "Adam"}'


# nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 1,1,1,1,1,1

# name="fed_isic_supervised_scaffold_${model_name}_${seed}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated \
# -method scaffold \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed ${seed} \
# --amp \
# --opt_args '{"class": "Adam"}'


# nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 1,1,1,1,1,1