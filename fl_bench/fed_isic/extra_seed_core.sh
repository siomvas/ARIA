conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

model=$1
seed=$2
gpu=$3
method=$4
init=$5
config=$6
rounds=$7
steps=$8
num_clients=$9

model_name=${model}_${init}


name="fed_isic_${init}_${method}_${model_name}_${seed}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_${config} \
-method ${method} \
-num_rounds ${rounds} \
-lr 5e-4 \
--batch_size 64 \
-local_steps ${steps} \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'


nvflare simulator ${algorithms_dir}/fed_isic_${config} -w ${RESULT_ROOT}/${name} -n ${num_clients} -gpu ${gpu}