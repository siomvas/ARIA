declare -a arr=("densenet_121_pretrained_pt" "supervised"
                )

gpu=$1


for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source medmnist_gpu05_fedopt.sh ${arr[i]} ${arr[i+1]} 100 $gpu
    source medmnist_gpu05_fedopt.sh ${arr[i]} ${arr[i+1]} 0.5 $gpu
done