conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic/medmnist
RESULT_ROOT=/path/to/sharedscratch/extra/
algorithms_dir="${PWD}/job_configs"

# create log file
LOG_FILE="$(basename $0).log"
touch "$LOG_FILE"

model=$1
seed=$2
method=$3
init=ssl
alpha=100
model_name=${model}_${init}


name="${init}_${method}_${model_name}_${seed}"

# execute script and log any errors
if ! python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${method} \
-num_rounds 50 \
-alpha ${alpha} \
-lr 0.01 \
--batch_size 64 \
-local_steps 100 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
&> "$LOG_FILE"; then
    echo "Error occurred with the following arguments: $@" >> "$LOG_FILE"
    echo "Reason for error:" >> "$LOG_FILE"
    tail -n 1 "$LOG_FILE" >> "$LOG_FILE"
fi

# execute nvflare and log any errors
if ! nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,1,1 &>> "$LOG_FILE"; then
    echo "Error occurred with the following arguments: $@" >> "$LOG_FILE"
    echo "Reason for error:" >> "$LOG_FILE"
    tail -n 1 "$LOG_FILE" >> "$LOG_FILE"
fi





