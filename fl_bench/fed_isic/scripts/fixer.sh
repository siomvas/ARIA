conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

gpu=0



model_name=vit_b_16_ssl
init=ssl
seed=42
name="fix_fed_isic_${init}_central_${model_name}_${seed}"

python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_central \
-method fedavg \
-num_rounds 2 \
-lr 5e-4 \
--batch_size 64 \
-local_steps 2910 \
-model_name ${model_name} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}' \
--use_hyperion_dirs \

nvflare simulator ${algorithms_dir}/fed_isic_central -w ${RESULT_ROOT}/${name} -n 1 -gpu ${gpu}