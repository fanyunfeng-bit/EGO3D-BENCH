import torch
from datasets import load_dataset
import json
import os
import sys
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.eval import *
from utils.common import *

### fix the seed:
torch.manual_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--image_root", type=str, default='Ego3D-Bench/images')
    args = parser.parse_args()

    path = args.model_path
    model_name = args.model_name

    ## load model
    print(">>> Using HuggingFace AutoModel backend for inference...")
    device = "cuda"
    model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor_qwen = AutoProcessor.from_pretrained(args.model_path,use_fast=True)

    ## load dataset
    dataset = load_dataset("vbdai/Ego3D-Bench")['test']

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
            image_order = ['Front_Left','Front','Front_Right','Side_Right','Back_Right','Back_Left','Side_Left']
        else:
            image_order = list(image_path.keys())

        image_path_sorted = [os.path.join(args.image_root, image_path[img]) for img in image_order]

        messages = convert_to_qwen_input(question,image_path_sorted)
        # HuggingFace inference
        text = processor_qwen.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor_qwen(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model_qwen.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor_qwen.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

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
        eval_logs(save_path[category],multi_choice=multi_choice)
