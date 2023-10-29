declare -a arr=("resnet_18" "scratch" \
                "resnet_50" "scratch" \
                "vit_b_16" "scratch" \
                "swin_t" "scratch" \
                "wide_resnet_50_2" "scratch" \
                "densenet_121" "scratch" \
                "convnext_small" "scratch" \
                "efficientnet_v2_s" "scratch" \
                )

alpha=$1
gpu=$2

for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source medmnist_scaffold_one_model.sh ${arr[i]} ${arr[i+1]} ${alpha} ${gpu}
done