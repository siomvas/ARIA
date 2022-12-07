#!/bin/bash
#SBATCH -D /path/to/FL_bench/fl_bench/
#SBATCH -J fl
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=6             # Use 4 cores
#SBATCH --mem=32GB                    # CPU RAM
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.

#SBATCH --mail-type=ALL,TIME_LIMIT_90
#SBATCH --mail-user=user@example.org

flight env activate conda
conda activate flare

RESULT_ROOT=../tmp/nvflare/sim_cifar10

echo $PWD
echo $CUDA_VISIBLE_DEVICES
# split dataset
python3 ./utils/cifar10_helpers.py


algorithms_dir="${PWD}/cifar10/job_configs"

./utils/set_update_parameters.py -job=${algorithms_dir}/cifar10_fedavg --alpha 1 --lr 1e-4 --num_rounds 2 --epochs 2

nvflare simulator cifar10/job_configs/cifar10_fedavg -w ${RESULT_ROOT}/fedavg_alpha1 -n 8 --threads 8
# -gpu $CUDA_VISIBLE_DEVICES



# # central
# ./set_alpha.sh cifar10_central 0.0
# nvflare simulator job_configs/cifar10_central --workspace ${RESULT_ROOT}/central --threads 1 --n_clients 1

# # FedAvg
# ./set_alpha.sh cifar10_fedavg 1.0
# nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha1.0 --threads 8 --n_clients 8
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