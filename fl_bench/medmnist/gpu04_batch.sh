declare -a arr=("resnet_18_pretrained_pt" "supervised" \
                "resnet_50_pretrained_pt" "supervised" \
                "vit_b_16_pretrained_pt" "supervised" \
                "swin_t_pretrained_pt" "supervised" \
                "wide_resnet_50_2_pretrained_pt" "supervised" \
                "densenet_121_pretrained_pt" "supervised" \
                "convnext_small_pretrained_pt" "supervised" \
                "efficientnet_v2_s_pretrained_pt" "supervised" \
                )

gpu=$1


for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source medmnist_gpu04_fedopt.sh ${arr[i]} ${arr[i+1]} 100 $gpu
    source medmnist_gpu04_fedopt.sh ${arr[i]} ${arr[i+1]} 0.5 $gpu
done