import torch
from transformers import  AutoModel, AutoTokenizer
import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.eval import *
from utils.internvl3_utils import *
from datasets import load_dataset
import argparse

torch.manual_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--image_root", type=str, default='Ego3D-Bench/images')
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for inference")
    args = parser.parse_args()

    path = args.model_path
    model_name = args.model_name

    print(">>> Using HuggingFace AutoModel backend for inference...")
    device = "cuda"
    device_map = split_model(path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype="auto",
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    ## load dataset
    dataset = load_dataset("/home/ma-user/work/.cache/huggingface/datasets/vbdai___ego3_d-bench/default/0.0.0/60b13042271d14c223a980a773dadc58543a2d03")["test"]
    # dataset = load_dataset("vbdai/Ego3D-Bench")['test']

    ## output file
    processsed={}
    save_path={}
    limit={}
    idx = {}
    if not os.path.exists(f"logs/{model_name}"):
        os.mkdir(f"logs/{model_name}")
    for category in set(dataset['category']):
        save_path[category] = f"logs/{model_name}/{category}.jsonl"
        processsed[category] = 0
        idx[category] = 0
        limit[category] = 5
        if os.path.exists(save_path[category]):
            with open(save_path[category]) as f:
                processsed[category] = sum(1 for _ in f)
            print(f"Resuming from {processsed[category]} processed samples in  {category} sub-category.")
    
    for sample in tqdm(dataset):

        idx[sample['category']]+=1
        # skip processed samples
        if idx[sample['category']] < processsed[sample['category']] or idx[sample['category']] > limit[sample['category']]:
            continue
        
        image_path = sample['images']
        question = sample['question']
        answer = sample['answer']
        options = sample['options']

        if options:
            for option in options:
                question += '\n' + option
        
        if sample['category'] in ['Ego_Centric_Absolute_Distance','Object_Centric_Absolute_Distance']:
            question += "\nOutput the thinking process in <think> </think> and final answer (number only) in <answer> </answer> tags."
        elif sample['category'] in ['Ego_Centric_Relative_Distance','Ego_Centric_Motion_Reasoning','Object_Centric_Motion_Reasoning']:
            question += "\nOutput the thinking process in <think> </think> and final answer (yes or no) in <answer> </answer> tags."
        else:
            question += "\nOutput the thinking process in <think> </think> and final answer (only the letter of the choice) in <answer> </answer> tags."

        # determine image order
        if sample['source'] == 'nuscenes':
            image_order = ['Front_Left','Front','Front_Right','Back_Right','Back','Back_Left']
        elif sample['source'] == 'waymo':
            image_order = ['Front','Front_Left','Side_Left','Front_Right','Side_Right']
        elif sample['source'] == 'argoverse':
            image_order = ['Front_Left','Front','Front_Right','Right','Back_Right','Back_Left','Left']
        else:
            image_order = list(image_path.keys())

        image_path_sorted = [os.path.join(args.image_root, image_path[img]) for img in image_order]

        pixel_values, num_patches_list = prepare_images_internvl(image_path_sorted)

        if args.use_vllm:
            # vLLM inference
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            outputs = model.generate([prompt], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
        else:
            # HuggingFace inference
            response, history = model.chat(
                tokenizer, pixel_values, question, generation_config,
                num_patches_list=num_patches_list,
                history=None, return_history=True
            )

        print(response)
        response_processed = response.split('<answer>')[-1].split('</answer>')[0]
        response_processed = response_processed.replace('\n','').strip()

        with open(save_path[sample['category']], "a") as file_out:
            file_out.write(json.dumps({
                'Question': question,
                'Question_Type': 'multi_choice' if options else 'exact_number',
                'Pred': response,
                'Processed_Pred': response_processed,
                'GT': sample['answer']
            }) + "\n")
    
    ### Evaluate the stored log:
    for category in set(dataset['category']):
        save_path[category] = f"logs/{model_name}/{category}.jsonl"
        
        ## We have two categories  with exact number as output and the rest if categories are multi-choice
        if category in ['Ego_Centric_Absolute_Distance','Object_Centric_Absolute_Distance']:
            multi_choice=False
        else:
            multi_choice=True
        print('#####',category)
        eval_logs(save_path[category],multi_choice=multi_choice)