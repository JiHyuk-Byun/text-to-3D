
parent_dir='/home/work/dataset/gobjaverse/gobjaverse_alignment' #./example_material/qwen2_example' #gobjaverse/gobjaverse_alignment/{921~1945}/5002955/campos_512_v2/{view_number:05d}/view_number:05d.png
out_file='/home/work/dataset/gobjaverse/qwen2_caption/test5/' # If defined, captions will be saved in a single file.
#921~1945, some empty
s_index=921
e_index=1522
gpu='0,1'
num_processes=2
batch_size=1
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
    model_args="--batch_n $index --parent_dir ${in_dir} --model_type $model_type --n_view $n_view --single_pkl True --out_file $out_file"
    #CUDA_VISIBLE_DEVICES=$gpu python3 batch_caption_qwen2.py --batch_n $index --parent_dir ${in_dir} --model_type $model_type --n_view $n_view --single_pkl True --out_file $out_file --batch_size $batch_size

    CUDA_VISIBLE_DEVICES=$gpu python3 -m accelerate.commands.launch --num_processes=${num_processes} --main_process_port=29400 -m captioners.fast_caption_qwen2 ${model_args}
#    python3 caption_clip_gpt_gob.py --parent_dir ${in_dir} --gpt_type $gpt_type --n_view $n_view

done

echo "\nAll data is captioned!"
