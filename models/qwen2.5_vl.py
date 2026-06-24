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
    parser.add_argument("--attn", type=str, default="flash_attention_2",
                        help="Attention backend: flash_attention_2 | sdpa | eager")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only process this many samples per category (smoke test)")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Optional cap on per-image vision pixels (processor default 12.8M if unset). "
                             "Not needed for memory thanks to the last-token lm_head patch below; kept as an escape hatch.")
    parser.add_argument("--min_pixels", type=int, default=None,
                        help="Optional lower bound on per-image vision pixels (processor default if unset).")
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
        attn_implementation=args.attn,
    )

    # --- Memory fix: only run lm_head on the LAST token. ---
    # This transformers version's Qwen2_5_VL.forward has no logits_to_keep arg and runs
    # lm_head over the FULL prompt. With 5-7 high-res views (~20k vision tokens) that logits
    # tensor is ~6 GiB and OOMs on a 24 GB GPU. Greedy generation only ever uses the last
    # position, so slicing here is numerically identical (no image downscaling needed).
    import torch.nn as nn
    class _LastTokenHead(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
        def forward(self, hidden_states):
            if hidden_states.dim() == 3 and hidden_states.size(1) > 1:
                hidden_states = hidden_states[:, -1:, :]
            return self.head(hidden_states)
    model_qwen.lm_head = _LastTokenHead(model_qwen.lm_head)

    _proc_kwargs = {"use_fast": True}
    if args.max_pixels is not None:
        _proc_kwargs["max_pixels"] = args.max_pixels
    if args.min_pixels is not None:
        _proc_kwargs["min_pixels"] = args.min_pixels
    processor_qwen = AutoProcessor.from_pretrained(args.model_path, **_proc_kwargs)

    ## load dataset
    dataset = load_dataset("vbdai/Ego3D-Bench")['test']

    ## output file
    processsed={}
    save_path={}
    # limit={}
    idx = {}
    os.makedirs(f"logs/{model_name}", exist_ok=True)
    for category in set(dataset['category']):
        save_path[category] = f"logs/{model_name}/{category}.jsonl"
        processsed[category] = 0
        idx[category] = 0
        # limit[category] = 5
        if os.path.exists(save_path[category]):
            with open(save_path[category]) as f:
                processsed[category] = sum(1 for _ in f)
            print(f"Resuming from {processsed[category]} processed samples in  {category} sub-category.")
    
    for sample in tqdm(dataset):
        idx[sample['category']]+=1
        # skip already-processed samples (resume): skip exactly `processsed` per category
        if idx[sample['category']] <= processsed[sample['category']]: # or idx[sample['category']] > limit[sample['category']]:
            continue
        # smoke-test cap: only the first --limit samples per category
        if args.limit is not None and idx[sample['category']] > args.limit:
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
        else:
            # NOTE: all non-exact-number categories are multiple-choice with a LETTER ground
            # truth (A/B/...). Upstream told relative-distance / motion-reasoning to answer
            # "yes or no", which never matches the letter GT -> spurious 0.0 accuracy. Ask for
            # the letter for every multiple-choice category (matches GT and the ego3dvlm script).
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
