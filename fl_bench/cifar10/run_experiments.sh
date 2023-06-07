#!/bin/bash
#SBATCH -D /path/to/FL_bench
#SBATCH -J fl
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=4             # Use 4 cores
#SBATCH --mem=32GB                    # CPU RAM
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.

#SBATCH --mail-type=ALL,TIME_LIMIT_90
#SBATCH --mail-user=user@example.org

flight env activate conda
conda activate flare

RESULT_ROOT=/path/to/archive/tmp/tofix

# split dataset
python3 fl_bench/utils/cifar10_helpers.py


algorithms_dir="${PWD}/fl_bench/cifar10/job_configs"

# python3 ${PWD}/fl_bench/utils/set_update_parameters.py -job=${algorithms_dir}/cifar10_fedavg --alpha 0.6 --lr 1e-2 --num_rounds 50 --epochs 4

nvflare simulator ${algorithms_dir}/cifar10_fedavg -w ${RESULT_ROOT}/checking_1gpu -n 10 -t 10 -gpu 0



# central
./set_alpha.sh cifar10_central 0.0
nvflare simulator job_configs/cifar10_central --workspace ${RESULT_ROOT}/central --threads 1 --n_clients 1

# FedAvg
./set_alpha.sh cifar10_fedavg 1.0
nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha1.0 --threads 8 --n_clients 8
# ./set_alpha.sh cifar10_fedavg 0.5
# nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha0.5 --threads 8 --n_clients 8


# # FedProx
# ./set_alpha.sh cifar10_fedprox 0.1
# nvflare simulator job_configs/cifar10_fedprox --workspace ${RESULT_ROOT}/fedprox_alpha0.1 --threads 8 --n_clients 8

# # FedOpt
# ./set_alpha.sh cifar10_fedopt 0.1
# nvflare simulator job_configs/cifar10_fedopt --workspace ${RESULT_ROOT}/fedopt_alpha0.1 --threads 8 --n_clients 8

# # SCAFFOLD
# ./set_alpha.sh cifar10_scaffold 0.1
# nvflare simulator job_configs/cifar10_scaffold --workspace ${RESULT_ROOT}/scaffold_alpha0.1 --threads 8 --n_clients 8