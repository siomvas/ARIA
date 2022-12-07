#!/usr/bin/env bash
algorithms_dir="${PWD}/cifar10/job_configs"
config=$1
alpha=$2

python3 ./utils/set_alpha.py --job=${algorithms_dir}/${config} --alpha=${alpha}
