conda activate flare
cd /path/to/FL_bench/fl_bench/medmnist
RESULT_ROOT=/path/to/localscratch/MedMNIST/
algorithms_dir="${PWD}/job_configs"
name="direct_session_new"

# Central
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/medmnist_central \
-method fedavg \
-alpha -2 \
-num_rounds 2 \
-lr 0.01 \
-local_steps 4 \
-model_name efficientnetv2_s_pretrained_pt \
-seed 222

nvflare simulator ${algorithms_dir}/medmnist_central -w ${RESULT_ROOT}/${name} -n 1 -gpu 0