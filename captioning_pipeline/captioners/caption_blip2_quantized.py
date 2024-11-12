import torch
from PIL import Image
import glob
import pickle as pkl
from tqdm import tqdm
import os
import argparse
# pip install accelerate bitsandbytes
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type = str, default='./example_material') #gobjaverse/gobjaverse_alignment/{921~1945}
    parser.add_argument("--model_type", type = str, default='pretrain_flant5xxl', choices=['pretrain_flant5xxl', 'pretrain_flant5xl'])
    parser.add_argument("--use_qa", action="store_true")
    parser.add_argument("--n_view", type = int, default=30)
    return parser.parse_args()

def main(view_number):

    args = parse_args()
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    all_output = {}

    name = 'blip2_t5'
    model_type = args.model_type

    outfilename = f'{args.parent_dir}/gobjaverse_captions_view{view_number}.pkl' # 300 images' captions
    # gobjaverse/gobjaverse_alignment/{921~1945}/ +  /5002955/campos_512_v2/{view_number:05d}/view_number:05d.png
    infolder = f'{args.parent_dir}/*/campos_512_v*/{view_number:05d}/{view_number:05d}.png'#현재 batch내의 모든 view_number들 #Cap3D_imgs/Cap3D_imgs_view{view_number}/*.png' 
    print(infolder)
    #print(infolder)
    if os.path.exists(outfilename):
        with open(outfilename, 'rb') as f:
            all_output = pkl.load(f)

    print("number of annotations so far",len(all_output))
    
    accelerator = Accelerator()
    # Load model
    #model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True)
    ct = 0

    model = accelerator.prepare(model)
    # Load images
    all_files = glob.glob(infolder)
    
    all_imgs = [x for x in all_files if ".png" in x.split("/")[-1]] # Check only png files
    print("len of .png", len(all_imgs)) 

    all_imgs = [x for x in all_imgs if x.split("/")[-4] not in all_output] # Check only new files
    print("len of new", len(all_imgs))

    for filename in tqdm(all_imgs):
        # skip the images we have already generated captions
        if os.path.exists(outfilename):
                if os.path.dirname(filename).split('/')[-3] in all_output.keys(): # e.g. uid: 5002955
                    continue
        try:
            raw_image = Image.open(filename).convert("RGB")
        except:
            print("file not work skipping", filename)
            continue

        # tokenize the images and prompt
        if args.use_qa:
            prompt = "Question: what object is in this image? Answer:"
            inputs = processor(raw_image, prompt, return_tensors="pt").to(accelerator.device, torch.float16)
            object = processor.decode(model.generate(**inputs)[0], skip_special_tokens=True)

            full_prompt = "Question: what is the structure and geometry of this %s?" % object
            inputs = processor(raw_image, full_prompt, return_tensors="pt").to(accelerator.device, torch.float16)
            outputs = model.generate(**inputs, num_return_sequences=5, do_sample=True)#.to(accelerator.device, torch.float16)
            x = [processor.decode(output, skip_special_tokens=True) for output in outputs]
        else:
            inputs = processor(images=raw_image, return_tensors="pt").to(accelerator.device, torch.float16)
            outputs = model.generate(**inputs, num_return_sequences=5, do_sample=True)#.to(accelerator.device, torch.float16)
            x = [processor.decode(output, skip_special_tokens=True) for output in outputs]

        all_output[os.path.dirname(filename).split('/')[-4]+os.path.dirname(filename).split('/')[-3]] = [z for z in x]
        
        if ct < 10 or (ct % 100 == 0 and ct < 1000) or (ct % 1000 == 0 and ct < 10000) or ct % 10000 == 0:
            print(filename)
            print([z for z in x])

            with open(outfilename, 'wb') as f:
                pkl.dump(all_output, f)
            
        ct += 1

    with open(outfilename, 'wb') as f:
        pkl.dump(all_output, f)

if __name__ == "__main__":
    args = parse_args()
    n_view = args.n_view
    for i in range(n_view):
        main(view_number=i)
