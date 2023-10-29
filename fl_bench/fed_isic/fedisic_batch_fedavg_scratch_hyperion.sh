declare -a arr=("resnet_18" "scratch" \
                "resnet_50" "scratch" \
                "vit_b_16" "scratch" \
                "swin_t" "scratch" \
                "wide_resnet_50_2" "scratch" \
                "densenet_121" "scratch" \
                "convnext_small" "scratch" \
                "efficientnet_v2_s" "scratch"
                )



for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source fedisic_fedavg_scratch_model_hyperion.sh ${arr[i]} ${arr[i+1]} 0,0,0,1,1,1
done