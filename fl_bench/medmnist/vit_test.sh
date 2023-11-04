conda activate flare
cd /path/to/FL_bench/fl_bench/medmnist
RESULT_ROOT=/path/to/localscratch/MedMNIST/
algorithms_dir="${PWD}/job_configs"

model_name=vit_b_16_pretrained_ssl
gpu=$1

############################################################################################################
name="medmnist_central_ssl_${model_name}_test"


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

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu ${gpu}