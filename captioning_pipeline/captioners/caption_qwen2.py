import torch
from PIL import Image
import glob
import _pickle as pkl
from tqdm import tqdm
import os
import argparse
# pip install accelerate bitsandbytes
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from accelerate import Accelerator
import time

models = {'72b_instruct': '-72B-Instruct',
        '72b_instruct_awq': '-72B-Instruct-AWQ',
        '72b_gptq_int4': '-72B-Instruct-GPTQ-Int4',}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_n", type = int)
    parser.add_argument("--parent_dir", type = str, default='./example_material') #gobjaverse/gobjaverse_alignment/{921~1945}
    parser.add_argument("--model_name", type = str, default='blip2_t5', choices=['blip2_t5', 'qwen2_vl'])
    parser.add_argument("--model_type", type = str, default='pretrain_flant5xxl', choices=['72b_instruct', '72b_instruct_awq', '72b_gptq_int4'])
    parser.add_argument("--out_file", type = str, default='./example_material')
    parser.add_argument("--single_pkl", type = bool, default=False) # If defined, captions will be saved in a single file.
    parser.add_argument("--use_qa", action="store_true")
    parser.add_argument("--n_view", type = int, default=30)
    return parser.parse_args()

def main():

    args = parse_args()
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    all_output = {}
    
    #####
    name = args.model_name
    model_type = "Qwen/Qwen2-VL" + models[args.model_type]
    ct = 0
    n_view = args.n_view
    print("model type: ", model_type)
    outfilename = args.out_file + f'/gpu1_captions_qwen2_{args.batch_n}_v6.pkl' if args.single_pkl else f'{args.parent_dir}/gobjaverse_captions.pkl' # 300 images' captions
    # gobjaverse/gobjaverse_alignment/{921~1945}/ +  /5002955/campos_512_v2/{view_number:05d}/view_number:05d.png
    infolder = f'{args.parent_dir}/*/campos_512_v*/' #gobjaverse/gobjaverse_alignment/{921~1945}/*/campos_512_v*
    #####
    
    # Load out_file
    if os.path.exists(outfilename):
        with open(outfilename, 'rb') as f:
            all_output = pkl.load(f)

    print("number of annotations so far",len(all_output))
    
    accelerator = Accelerator()
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_type,#-GPTQ-Int4",
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_type)

    model = accelerator.prepare(model)

    # Load image files which is only new ones.
    all_files = glob.glob(infolder)
    all_files = [x for x in all_files if os.path.join(x.split("/")[-4], x.split("/")[-3]) not in all_output.keys()] # Check only new files
    print("Len of new: ", len(all_files))
    all_files = sorted(all_files, key=lambda x: int(x.split('/')[-3]))

    # Prompt
    prompt = f"Below are images of a 3D object captured from various angles. Write a detailed caption in one or two sentences that describes the object's main features, focusing on geometric details and providing detailed color descriptions. For example:\
            \n- 'A spherical object with a diameter of approximately 10 centimeters, featuring a smooth surface made of glossy emerald-green glass with subtle swirls of lighter green, and adorned with intricate gold filigree patterns encircling its equator.'\
            \n- 'A muscular humanoid figure showcasing a broad chest and defined musculature. The skin mimics a rough stone texture in varying shades of gray and brown, giving the appearance of a living rock creature. The face features deep-set eyes with glowing orange irises and an angular jawline. Sharp, geometric plates protrude from the shoulders and back, adding a rugged and geometric aesthetic.'\
            \n- 'An angular mechanical structure with a central rectangular body and large panels that fan out symmetrically, featuring a dark metallic color scheme of gray and black, with green accents on some panels and red details along the edges for visual contrast. Two cylindrical supports extend downward, and narrow rectangular slits enhance its industrial and fortified appearance.'\
            \nNow, please write a similar caption for the provided images, including shape, dimensions, structural details, unique geometric characteristics, and detailed descriptions of colors. **Avoid describing the background, surface, and posture.** The caption should be:"
    
    if os.path.exists(outfilename):
        assert all_output["prompt"] == prompt, f"Current version of prompt: \n {prompt} \n is different from the given out filename's prompt: \n {all_output['prompt']}.\n Please revise the out filename."
    else:
        all_output["prompt"] = prompt

    # Per image index: gobjaverse/gobjaverse_alignment/{921~1945}/*/campos_512_v*
    for img_dir in tqdm(all_files):
        # If already exists, pass
        if os.path.exists(outfilename):
            if os.path.join(img_dir.split("/")[-4], img_dir.split("/")[-3]) in all_output.keys(): # e.g. uid: 5002955
                continue

        # Load image paths
        print(f"Now {img_dir.split('/')[-2]} is processing...")

        download_path = os.path.join(img_dir, "*")
        infiles = glob.glob(download_path)
        infiles = sorted(infiles, key=lambda x: int(x.split('/')[-1]))
        
        # Check the number of entire views.
        entire_files = len(infiles)
        assert entire_files == 40 or 52, f"Invalid number of file with {entire_files} views."
        round_view = 24 if entire_files == 40 else 36
        step = int(round_view / n_view)

        # Prepare inputs
        content = []
        # Load text
        print("Input prompt:")
        print(prompt)
        content.append({"type": "text", "text": prompt})

        # Load images with given step.
        print("Input images:")
        for i, view in enumerate(range(0, round_view, step)):
            file_path = os.path.join(infiles[view], f"{view:05d}.png")
            print(file_path)
            content.append({"type": "image", "image": file_path})
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Tokenize inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to("cuda")

        # Generate outputs.
        s_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        e_time = time.time()
        print("gen time: ", e_time - s_time)

        # Decode outputs
        s_time = time.time()
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        e_time = time.time()
        print("decode time: ", e_time - s_time)

        # Recode the caption.
        all_output[os.path.join(img_dir.split("/")[-4], img_dir.split("/")[-3])] = [output_text for output_text in output_texts]
        
        print([output_text for output_text in output_texts])
        with open(outfilename, 'wb') as f:
            pkl.dump(all_output, f)

        # if ct < 10 or (ct % 100 == 0 and ct < 1000) or (ct % 1000 == 0 and ct < 10000) or ct % 10000 == 0:
        #     print([output_text for output_text in output_texts])

        #     with open(outfilename, 'wb') as f:
        #         pkl.dump(all_output, f)


        # ct += 1

    with open(outfilename, 'wb') as f:
        pkl.dump(all_output, f)

if __name__ == "__main__":
    
    main()
