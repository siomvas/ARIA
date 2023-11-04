declare -a arr=("central" "scratch" "central" 2 8730 1 \
                "fedavg" "scratch" "federated" 20 600 6 \
                "fedopt" "scratch" "federated" 2 600 6 \
                "scaffold" "scratch" "federated" 20 600 6 \
                "central" "supervised" "central" 2 2910 1 \
                "fedavg" "supervised" "federated" 20 200 6 \
                "fedopt" "supervised" "federated" 20 200 6 \
                "scaffold" "supervised" "federated" 20 200 6 \
                "central" "ssl" "central" 2 2910 1 \
                "fedavg" "ssl" "federated" 20 200 6 \
                "fedopt" "ssl" "federated" 20 200 6 \
                "scaffold" "ssl" "federated" 20 200 6 \
                )

model=$1
gpu=$2
seed=$3


for i in "${!arr[@]}"
do
    if (( i % 6 != 0 )); then
        continue
    fi
    source extra_seed_core.sh ${model} ${seed} ${gpu} ${arr[i]} ${arr[i+1]} ${arr[i+2]} ${arr[i+3]} ${arr[i+4]} ${arr[i+5]}
done