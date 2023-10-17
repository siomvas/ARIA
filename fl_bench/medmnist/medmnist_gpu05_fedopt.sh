conda activate flare
cd /path/to/FL_bench/fl_bench/medmnist
RESULT_ROOT=/path/to/localscratch/MedMNIST/
algorithms_dir="${PWD}/job_configs/"

model_name=$1
init=$2
alpha=$3
agg=fedopt
gpu=$4

name="organamnist_alpha=${alpha}_${init}_${agg}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 50 \
-lr 0.01 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed 111 \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}' \
# --opt_args '{"class": "Adam", "weight_decay": 1e-5}'

# OVERWRITE CHECK
if [ -d "${RESULT_ROOT}/${name}" ]; then
    echo "Directory ${RESULT_ROOT}/${name} already exists."
    read -p "Do you want to overwrite it? (Y/N) " choice

    if [[ $choice == "Y" || $choice == "y" ]]; then
        echo "Continuing..."
    else
        echo "Aborting..."
        return 1
    fi

else
    echo "Directory ${RESULT_ROOT}/${name} does not exist."
fi

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu ${gpu},${gpu},${gpu},${gpu}