declare -a arr=("resnet_18_pretrained_pt" "supervised" \
                "swin_t_pretrained_pt" "supervised" \
                "efficientnet_v2_s_pretrained_pt" "supervised" \
                )



for i in "${!arr[@]}"
do
    if (( i % 2 != 0 )); then
        continue
    fi
    source fedisic_fedopt_supervised_model_hyperion.sh ${arr[i]} ${arr[i+1]} 1,1,1,3,3,3
done