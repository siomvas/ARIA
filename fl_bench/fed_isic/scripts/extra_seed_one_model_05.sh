declare -a arr=("scaffold" "supervised" "federated" 20 200 6 \
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