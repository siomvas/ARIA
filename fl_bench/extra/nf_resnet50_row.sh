#!/bin/bash
#SBATCH -D /path/to/FL_bench/fl_bench/extra
#SBATCH -J organ_nf
#SBATCH --partition=gengpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6             
#SBATCH --mem=40GB                    # CPU RAM
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.


source /opt/apps/flight/env/conda+default/etc/profile.d/conda.sh
conda activate flare
RESULT_ROOT=/path/to/sharedscratch/extra/


# MEDMNIST SECTION

algorithms_dir="/path/to/FL_bench/fl_bench/medmnist/job_configs/hyperion_1"

RESULT_ROOT=/path/to/sharedscratch/extra/

for method in fedavg fedopt scaffold; do
    for model_name in nf_resnet50_scratch nf_resnet50_supervised nf_resnet50_ssl; do
        for seed in 111 222; do
            name="medmnist_${method}_${model_name}_${seed}"
            python3 /path/to/FL_bench/fl_bench/medmnist/src/setup.py \
                -job ${algorithms_dir}/medmnist_federated \
                -method ${method} \
                -alpha 100 \
                -num_rounds 50 \
                -lr 0.01 \
                --batch_size 64 \
                -local_steps 200 \
                -model_name ${model_name} \
                -seed ${seed} \
                --amp \
                --use_hyperion_dirs \
                --opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'
            nvflare simulator ${algorithms_dir}/medmnist_federated -w ${RESULT_ROOT}/${name} -n 4 -gpu 0,0,1,1
        done
    done
done
