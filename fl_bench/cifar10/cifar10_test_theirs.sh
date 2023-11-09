conda activate flare
cd /path/to/FL_bench/fl_bench/cifar10
RESULT_ROOT=/path/to/localscratch/Cifar_10/
algorithms_dir="${PWD}/job_configs"

alpha=100
init=supervised
agg=fedavg
seed=42
model=$1
config=cifar10_federated_theirs

name="${config}_${init}_${model}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/${config} \
-method ${agg} \
-alpha ${alpha} \
-num_rounds 100 \
-lr 0.01 \
--batch_size 32 \
-local_steps 282 \
-model_name ${model}_${init} \
-seed ${seed} \
--amp \


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

nvflare simulator ${algorithms_dir}/${config} -w ${RESULT_ROOT}/${name} -n 5 -gpu 0,0,1,1,1
