# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: June 23, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

import os
import openai
from openai import OpenAI
import pickle
import torch
import clip
from PIL import Image
from torch.nn import CosineSimilarity
import csv
import argparse
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", type = str, default='./example_material')#gobjaverse/gobjaverse_alignment/{921~1945}
parser.add_argument('--gpt_type', type = str, default='gpt4o', choices=['gpt4o', 'gpt3.5'])
parser.add_argument('--n_view', type = int, default=30)
args = parser.parse_args()


# set up API key
client = OpenAI()

# set up CLIP
cos = CosineSimilarity(dim=1, eps=1e-6)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# set up GPT4
def summarize_captions_gpt4(text):
    prompt = f"Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows: '{text}'. Avoid describing background, surface, and posture. The caption should be:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            top_p = 0.2,
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            return "No response was received from gpt4o api."
    except Exception as e:
        return f"An error occurred: {str(e)}"

    return response.choices[0].message.content

# set up GPT3.5
def summarize_captions_gpt35(text):
    prompt = f"Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows: '{text}'. Avoid describing background, surface, and posture. The caption should be:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            top_p = 0.2,
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            return "No response was received from gpt4 api."
    except Exception as e:
        return f"An error occurred: {str(e)}"

    return response.choices[0].message.content


# load captions for each view (n views in total)
caps = []
n_view = args.n_view

for i in range(n_view):
    caps.append(pickle.load(open(os.path.join(args.parent_dir, 'gobjaverse_captions_view'+str(i)+'.pkl'), 'rb'))) # 원래는 view > key = uid, value = caption

names = []
for i in range(n_view):
    names.append(set([name for name in caps[i].keys()])) # 5002955 ...

uids = list(reduce(set.intersection, names))
print(uids)
#uids = [n for name in names for n in name] #list(reduce(set.intersection, names)) 원래는 각 uids > n_view의 구조로 저장되어 있어야 하는데, 여기서는 uid가 1개니깐.

# please remove existing uids 

# change 'w' to 'a' if you want to append to an existing csv file
output_csv = open(os.path.join(args.parent_dir, 'gobjaverse_captions_final.csv'), 'w')
writer = csv.writer(output_csv)

# TREE: gobjaverse/gobjaverse_alignment/{921~1945}/ +  /5002955/campos_512_v2/{view_number:05d}/view_number:05d.png
print('############begin to generate final captions############')
for i, cur_uid in enumerate(uids):
    cur_captions = []
    cur_final_caption = ''

    # for each view, choose the caption with the highest similarity score
    # run 8 times to get 8 captions, select only 1 caption among 5 captions per i
    for k in range(n_view):
        
        
        img_path = os.path.join(args.parent_dir, cur_uid, f'campos_512_v1', '%05d'%k, '%05d.png'%k)
        if not os.path.exists(img_path):
            img_path =  os.path.join(args.parent_dir, cur_uid, f'campos_512_v2', '%05d'%k, '%05d.png'%k)

        assert os.path.exists(img_path), f"{img_path} doesn't exist."

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        cur_caption = caps[k][f"{cur_uid}"]
        text = clip.tokenize(cur_caption).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        score = cos(image_features, text_features)
        # 5개 캡션중 가장 clip score가 높은 것만 저장.
        if k == n_view-1:
            cur_final_caption += cur_caption[torch.argmax(score)]
        else:
            cur_final_caption += cur_caption[torch.argmax(score)] + ', '

    print(cur_final_caption)

        # sometimes, OpenAI API will return an error, so we need to try again until it works
    while True:
        if args.gpt_type == 'gpt4o':
            summary = summarize_captions_gpt4(cur_final_caption)
        elif args.gpt_type == 'gpt3.5':
            summary = summarize_captions_gpt35(cur_final_caption)

        if 'An error occurred' not in summary:
            break

    print(f"summarized into: {summary}")
    #write to csv
    writer.writerow([cur_uid, summary])
    if (i)% 1000 == 0:
        output_csv.flush()
        os.fsync(output_csv.fileno())

output_csv.close()


