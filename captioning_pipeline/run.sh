parent_dir='./example_material/gobjaverse_example' #gobjaverse/gobjaverse_alignment/{921~1945}/5002955/campos_512_v2/{view_number:05d}/view_number:05d.png
s_index=1000
e_index=1001

model_type='pretrain_flant5xl'
gpt_type='gpt4o'
n_view=40

total=$((e_index - s_index + 1))
count=0
for index in $(seq $s_index $e_index) # batch당 300개
do 
    count=$((count + 1))
    in_dir="${parent_dir}/${index}"

    progress=$((count * 100 / total))
    echo -ne "Processing Batch: $in_dir ... $count / $total % Done.\r"

    python3 caption_blip2_quantized.py --parent_dir ${in_dir} --model_type $model_type --n_view $n_view

    python3 caption_clip_gpt_gob.py --parent_dir ${in_dir} --gpt_type $gpt_type --n_view $n_view


done

echo -e "\nAll data is captioned!"