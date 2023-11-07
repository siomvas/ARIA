conda activate flare
cd /path/to/FL_bench/fl_bench/cifar10
RESULT_ROOT=/path/to/localscratch/Cifar_10/
algorithms_dir="${PWD}/job_configs"

alpha=0.2
init=supervised
agg=fedavg
seed=42
model=$1
config=cifar10_central

name="${config}_${init}_${model}"

# 100 rounds of 100 steps each
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/${config} \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 100 \
-lr 0.01 \
--batch_size 64 \
-local_steps 100 \
-model_name ${model}_${init} \
-seed ${seed} \
--amp \
--opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 5e-4}' \


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

nvflare simulator ${algorithms_dir}/${config} -w ${RESULT_ROOT}/${name} -n 1 -gpu 1
