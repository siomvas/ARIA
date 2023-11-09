conda activate flare
cd /path/to/FL_bench/fl_bench/medmnist
RESULT_ROOT=/path/to/sharedscratch/extra/
algorithms_dir="${PWD}/job_configs"

# create log file
LOG_FILE="column_maker.log"
touch "$LOG_FILE"

init=$1
method=$2
seed=$3

models=("resnet_18" "resnet_50" "nf_resnet50" "wide_resnet_50_2" "vit_b_16" "densenet_121" "swin_t" "convnext_small" "efficientnet_v2_s")

for model in "${models[@]}"
do
    name="mnist01_${model}_${init}_${method}_${seed}"

    if ! python3 ${PWD}/src/setup.py \
        -job ${algorithms_dir}/medmnist_federated \
        -method ${method} \
        -alpha 0.1 \
        -num_rounds 50 \
        -lr 0.01 \
        --batch_size 64 \
        -local_steps 100 \
        -model_name ${model}_${init} \
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
done
