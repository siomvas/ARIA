declare -a arr=("resnet_18" "scratch" \
                "resnet_18_pretrained_pt" "supervised" \
                "resnet_50" "scratch" \
                "resnet_50_pretrained_pt" "supervised" \
                "vit_b_16" "scratch" \
                "vit_b_16_pretrained_pt" "supervised" \
                "swin_t" "scratch" \
                "swin_t_pretrained_pt" "supervised" \
                "wide_resnet_50_2" "scratch" \
                "wide_resnet_50_2_pretrained_pt" "supervised" \
                "densenet_121" "scratch" \
                "densenet_121_pretrained_pt" "supervised" \
                "convnext_small" "scratch" \
                "convnext_small_pretrained_pt" "supervised" \
                "efficientnet_v2_s" "scratch" \
                "efficientnet_v2_s_pretrained_pt" "supervised" \
                )



for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source medmnist_central_train_one_model.sh ${arr[i]} ${arr[i+1]}
done