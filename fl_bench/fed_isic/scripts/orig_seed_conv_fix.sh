conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

gpu=0,0,0,1,1,1

model_name=efficientnet_v2_s_supervised
init=supervised
seed=42
name="fix_fed_isic_${init}_fedopt_${model_name}_${seed}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method fedopt \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'

nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}



seed=24



name="fix_fed_isic_${init}_scaffold_${model_name}_${seed}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method scaffold \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'

nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}