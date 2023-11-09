#!/bin/bash
#SBATCH -D /path/to/FL_bench/fl_bench/extra
#SBATCH -J organ
#SBATCH --partition=gengpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6             
#SBATCH --mem=20GB                    # CPU RAM
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -e %x_%j.e                    # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o %x_%j.o                    # [%x with the job name], make sure 'results' folder exists.


source /opt/apps/flight/env/conda+default/etc/profile.d/conda.sh
conda activate flare

declare -a arr=("fedavg" 111 \
                "fedopt" 111 \
                "scaffold" 111 \
                "fedavg" 222 \
                "fedopt" 222 \
                "scaffold" 222 \
                )

model=$1


for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source mnist_ssl_core.sh ${model} ${arr[i+1]} ${arr[i]}
done