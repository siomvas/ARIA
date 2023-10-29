conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

model_name=$1
init=$2
gpu=$3

name="fed_isic_${init}_fedopt_${model_name}"


python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated \
-method fedopt \
-num_rounds 20 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 200 \
-model_name ${model_name} \
-seed 42 \
--opt_args '{"class": "Adam"}'

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

nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}

