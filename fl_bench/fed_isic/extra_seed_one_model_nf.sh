declare -a arr=("scaffold" "scratch" "federated" 20 600 6 \
                "central" "supervised" "central" 2 2910 1 \
                "fedopt" "supervised" "federated" 20 200 6 \
                "scaffold" "supervised" "federated" 20 200 6 \
                "fedavg" "ssl" "federated" 20 200 6 \
                "fedopt" "ssl" "federated" 20 200 6 \
                "scaffold" "ssl" "federated" 20 200 6 \
                )

model=nf_resnet50
gpu=1
seed=24


for i in "${!arr[@]}"
do
    if (( i % 6 != 0 )); then
        continue
    fi
    source extra_seed_core.sh ${model} ${seed} ${gpu} ${arr[i]} ${arr[i+1]} ${arr[i+2]} ${arr[i+3]} ${arr[i+4]} ${arr[i+5]}
done