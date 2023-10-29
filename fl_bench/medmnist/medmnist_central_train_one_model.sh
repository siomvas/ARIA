conda activate flare
cd /path/to/FL_bench/fl_bench/medmnist
RESULT_ROOT=/path/to/localscratch/MedMNIST/
algorithms_dir="${PWD}/job_configs"

model_name=$1
init=$2

name="organamnist_central_${init}_${model_name}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_central \
-method fedavg \
-alpha -1 \
-num_rounds 5 \
-lr 0.01 \
--batch_size 64 \
-local_steps 8000 \
-model_name ${model_name} \
-seed 111 \
--amp \
--use_hyperion_dirs \
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

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0