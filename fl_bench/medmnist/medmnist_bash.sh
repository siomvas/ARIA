conda activate flare
cd /path/to/FL_bench/fl_bench/medmnist
RESULT_ROOT=/path/to/localscratch/MedMNIST/
algorithms_dir="${PWD}/job_configs"
name="direct_session_fed_debug1"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_federated \
-method fedavg \
-alpha 1 \
-num_rounds 2 \
-lr 0.01 \
-local_steps 5 \
-model_name resnet_50 \
-seed 222

nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 1,1,1,1