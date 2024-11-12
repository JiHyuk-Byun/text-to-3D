
parent_dir='/home/work/dataset/gobjaverse/gobjaverse_alignment' #./example_material/qwen2_example' #gobjaverse/gobjaverse_alignment/{921~1945}/5002955/campos_512_v2/{view_number:05d}/view_number:05d.png
out_file='/home/work/dataset/gobjaverse/qwen2_caption' # If defined, captions will be saved in a single file.
#921~1945, some empty
s_index=921
e_index=1945
gpu=1
model_name='qwen2_vl'
model_type='72b_gptq_int4' #['72b_instruct', '72b_instruct_awq', '72b_gptq_int4']
gpt_type='gpt4o'
n_view=8

total=$(ls -l "$parent_dir" | grep ^d | wc -l)
count=0
for index in $(seq $s_index $e_index) # batch당 300개
do 
    count=$((count + 1))
    in_dir="${parent_dir}/${index}"

    if [ ! -e "$in_dir" ]; then
        continue
    fi
    progress=$((count * 100 / total))
    echo "Processing Batch: $in_dir ... ($count / $total % Done.)\r"

    CUDA_VISIBLE_DEVICES=$gpu python3 _caption_qwen2.py --batch_n $index --parent_dir ${in_dir} --model_type $model_type --n_view $n_view --single_pkl True --out_file $out_file

#    python3 caption_clip_gpt_gob.py --parent_dir ${in_dir} --gpt_type $gpt_type --n_view $n_view


done

echo "\nAll data is captioned!"
