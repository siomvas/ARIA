conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/new_models
algorithms_dir="${PWD}/job_configs"

seed=24
method=scaffold
init=supervised
steps=200
rounds=20

# model_name=nfnet_l0
# name="${init}_${method}_${model_name}_${seed}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated \
# -method ${method} \
# -num_rounds ${rounds} \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps ${steps} \
# -model_name ${model_name}_${init} \
# -seed ${seed} \
# --amp \
# --opt_args '{"class": "Adam"}'
# nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1



# model_name=eca_nfnet_l0
# name="${init}_${method}_${model_name}_${seed}"
# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated \
# -method ${method} \
# -num_rounds ${rounds} \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps ${steps} \
# -model_name ${model_name}_${init} \
# -seed ${seed} \
# --amp \
# --opt_args '{"class": "Adam"}'
# nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

# model_name=eca_nfnet_l1
# name="${init}_${method}_${model_name}_${seed}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated \
# -method ${method} \
# -num_rounds ${rounds} \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps ${steps} \
# -model_name ${model_name}_${init} \
# -seed ${seed} \
# --amp \
# --opt_args '{"class": "Adam"}'
# nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

# model_name=eca_nfnet_l2
# name="${init}_${method}_${model_name}_${seed}"

# python3 ${PWD}/src/setup.py \
# -job ${algorithms_dir}/fed_isic_federated \
# -method ${method} \
# -num_rounds ${rounds} \
# -lr 5e-4 \
# --batch_size 64 \
# -local_steps ${steps} \
# -model_name ${model_name}_${init} \
# -seed ${seed} \
# --amp \
# --opt_args '{"class": "Adam"}'

# nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1


model_name=gelu_nfnet_l0
name="${init}_${method}_${model_name}_${seed}"
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method ${method} \
-num_rounds ${rounds} \
-lr 5e-4 \
--batch_size 64 \
-local_steps ${steps} \
-model_name ${model_name}_${init} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1


model_name=gelu_eca_nfnet_l0
name="${init}_${method}_${model_name}_${seed}"
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method ${method} \
-num_rounds ${rounds} \
-lr 5e-4 \
--batch_size 64 \
-local_steps ${steps} \
-model_name ${model_name}_${init} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1

model_name=gelu_eca_nfnet_l1
name="${init}_${method}_${model_name}_${seed}"
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method ${method} \
-num_rounds ${rounds} \
-lr 5e-4 \
--batch_size 64 \
-local_steps ${steps} \
-model_name ${model_name}_${init} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1


model_name=gelu_eca_nfnet_l2
name="${init}_${method}_${model_name}_${seed}"
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method ${method} \
-num_rounds ${rounds} \
-lr 5e-4 \
--batch_size 64 \
-local_steps ${steps} \
-model_name ${model_name}_${init} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1