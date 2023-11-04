conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

gpu=$1

# model_name=densenet_121_pretrained_ssl
# init=ssl
# config=2

# name="fed_isic_${init}_scaffold_${model_name}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated_${config} \
# -method scaffold \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed 42 \
# --amp \
# --opt_args '{"class": "Adam"}' \
# --use_hyperion_dirs \

# nvflare simulator ${algorithms_dir}/fed_isic_federated_${config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}




# model_name=resnet_50_pretrained_ssl
# init=ssl
# config=2

# name="fed_isic_${init}_scaffold_${model_name}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated_${config} \
# -method scaffold \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed 42 \
# --amp \
# --opt_args '{"class": "Adam"}' \
# --use_hyperion_dirs \

# nvflare simulator ${algorithms_dir}/fed_isic_federated_${config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}

# model_name=swin_t_pretrained_ssl
# init=ssl
# config=2

# name="fed_isic_${init}_scaffold_${model_name}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated_${config} \
# -method scaffold \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed 42 \
# --amp \
# --opt_args '{"class": "Adam"}' \
# --use_hyperion_dirs \

# nvflare simulator ${algorithms_dir}/fed_isic_federated_${config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}



model_name=wide_resnet_50_2_pretrained_ssl
init=ssl
config=2

name="fed_isic_${init}_scaffold_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated_${config} \
-method scaffold \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed 42 \
--amp \
--opt_args '{"class": "Adam"}' \
--use_hyperion_dirs \

nvflare simulator ${algorithms_dir}/fed_isic_federated_${config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}

# model_name=convnext_small_pretrained_ssl
# init=ssl
# config=2

# name="fed_isic_${init}_scaffold_${model_name}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated_${config} \
# -method scaffold \
# -num_rounds 20 \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps 200 \
# -model_name ${model_name} \
# -seed 42 \
# --amp \
# --opt_args '{"class": "Adam"}' \
# --use_hyperion_dirs \

# nvflare simulator ${algorithms_dir}/fed_isic_federated_${config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}

model_name=vit_b_16_pretrained_ssl
init=ssl
config=2

name="fed_isic_${init}_scaffold_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated_${config} \
-method scaffold \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed 42 \
--amp \
--opt_args '{"class": "Adam"}' \
--use_hyperion_dirs \

nvflare simulator ${algorithms_dir}/fed_isic_federated_${config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}