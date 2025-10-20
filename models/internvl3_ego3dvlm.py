import torch
from PIL import Image,ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel, AutoTokenizer
import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.eval import *
from utils.common import *
from utils.cam_info import *
from utils.internvl3_utils import *
from datasets import load_dataset
import argparse

##### fix the seed:
torch.manual_seed(42)

device = "cuda"

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--rec_model_path",type=str)
    parser.add_argument("--depth_model_path",type=str)
    parser.add_argument("--wo_vision",action="store_true")
    parser.add_argument("--wo_vision_llm",action="store_true")
    parser.add_argument("--est_rt",action="store_true")
    parser.add_argument("--json_cogmap",action="store_true")
    parser.add_argument("--visual_cogmap",action="store_true")
    parser.add_argument("--image_root", type=str, default='Ego3D-Bench/images')
    args = parser.parse_args()
    path = args.model_path
    model_name = args.model_name

    debug=False
    if args.est_rt:
        est_rt = True
    else:
        est_rt = False


    if args.wo_vision_llm:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation = "flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        device_map = split_model(path)
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
            
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=False)  ## do_sample = False, to avoid sampling

    processor_depth = AutoImageProcessor.from_pretrained(args.depth_model_path)
    model_depth = AutoModelForDepthEstimation.from_pretrained(args.depth_model_path).to(device)

    processor_gdino = AutoProcessor.from_pretrained(args.rec_model_path)
    model_gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.rec_model_path).to(device)

    ## load dataset
    dataset = load_dataset("vbdai/Ego3D-Bench")['test']

    ## output file
    processsed={}
    save_path={}
    limit={}
    counter = {}
    if not os.path.exists(f"logs/{model_name}-ego3dvlm"):
        os.mkdir(f"logs/{model_name}-ego3dvlm")
    for category in set(dataset['category']):
        save_path[category] = f"logs/{model_name}-ego3dvlm/{category}.jsonl"
        processsed[category] = 0
        counter[category] = 0
        limit[category] = 5
        if os.path.exists(save_path[category]):
            with open(save_path[category]) as f:
                processsed[category] = sum(1 for _ in f)
            print(f"Resuming from {processsed[category]} processed samples in  {category} sub-category.")
    
    for sample in tqdm(dataset):

        if args.json_cogmap:
            cog_map={}
            cog_map["ego car"]={}
            cog_map["ego car"]["3D location"]=[0,0,0]
            cog_map["ego car"]["coordinate system"]={}
            cog_map["ego car"]["coordinate system"]["x"]="Positive x means right and negative x mean left to the ego car"
            cog_map["ego car"]["coordinate system"]["z"]="Positive z means infront of the ego car and negative z means behind the ego car"
            cog_map["detections"]=[]
        elif args.visual_cogmap:
            objects_xyz = []
            objects_names = []
        else:
            cog_map="\nEgo Car is at 3D location [x,y,z]=[0, 0, 0]." #, x is right(+)/left(-), y is up(+)/down(-), and z is forward(+)/backward(-).
            cog_map+="Positive x means right and negative x mean left to the ego car."
            cog_map+="Positive z means infront of the ego car and negative z means behind the ego car"

        counter[sample['category']]+=1
        # skip processed samples
        if counter[sample['category']] < processsed[sample['category']] or counter[sample['category']] > limit[sample['category']]:
            continue

        image_path = sample['images']
        question = sample['question']
        answer = sample['answer']
        options = sample['options']

        # determine image order
        if sample['source'] == 'nuscenes':
            image_order = ['Front_Left','Front','Front_Right','Back_Right','Back','Back_Left']
            scale = 0.86
        elif sample['source'] == 'waymo':
            image_order = ['Front','Front_Left','Side_Left','Front_Right','Side_Right']
            scale = 0.8
        elif sample['source'] == 'argoverse':
            image_order = ['Front_Left','Front','Front_Right','Side_Right','Back_Right','Back_Left','Side_Left']
            scale = 0.64

        image_path_sorted = [os.path.join(args.image_root, image_path[img]) for img in image_order]

        ### load camera paramters
        cams = camera_info(sample['source'], est_rt=est_rt)

        images = []
        cams_RT = []
        cams_K = []

        if debug:
            draws = []
            legends = []
            objects = []
            image_names = []

        for cams_idx in range(len(cams)):
            image = Image.open(image_path_sorted[cams_idx])
            images.append(image)
            cam_K = torch.tensor(cams[cams_idx]['intrinsic'])
            cam  = torch.eye(4)
            cam[:3,:3] = torch.tensor(cams[cams_idx]['rotation'])
            cam[:3,3] = torch.tensor(cams[cams_idx]['translation'])
            cams_RT.append(cam)
            cams_K.append(cam_K) 
            if debug:
                draw = ImageDraw.Draw(image)
                draws.append(draw)
                new_name=image_path_sorted[cams_idx].split('/')[-1]
                image_names.append(f"logs/{model_name}-ego3dvlm/{new_name}")

        cams_K=torch.stack(cams_K).to(device)
        cams_RT=torch.stack(cams_RT).to(device)        
        images = pad_images(images)
        target_sizes=[image.size[::-1] for _ in range(len(images))]


        if options:
            for option in options:
                question += '\n' + option
        
        if sample['category'] in ['Ego_Centric_Absolute_Distance','Object_Centric_Absolute_Distance']:
            question += "\nOutput the thinking process in <think> </think> and final answer (number ONLY) in <answer> </answer> tags."
        else:
            question += "\nOutput the thinking process in <think> </think> and final answer (ONLY the letter of the choice) in <answer> </answer> tags."


        #### We split the input prompt to sentences, and then apply REC to each senetenc
        question_trimed=strip_question(sample).split('.')
        question_trimed=[qs.replace('?','') for qs in question_trimed]
        text_labels=[question_trimed]*len(images)

        inputs_gdino = processor_gdino(images=images, text=text_labels, return_tensors="pt").to(device)

        inputs_depth = processor_depth(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_gdino = model_gdino(**inputs_gdino)
            outputs_depth = model_depth(**inputs_depth).predicted_depth
        
        
        predictions_gdino = processor_gdino.post_process_grounded_object_detection(
            outputs_gdino,
            inputs_gdino.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=target_sizes
        )

        # interpolate to original size
        prediction_depth = torch.nn.functional.interpolate(
            outputs_depth.unsqueeze(1),
            size=target_sizes[0], #image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()*scale
        
        
        world_coords = unproject(cams_K.float(), cams_RT.float(), prediction_depth.float()).cpu()    # (V, H, W, 3)
        bboxes_list = []

        for idx in range(len(predictions_gdino)):
            
            if debug:
                font = ImageFont.truetype("assets/Arial.ttf", size=50)  # You can use any TTF font file
                # Set text position and color
                text_position = (50, 50)  # Top-left corner
                text_color = (255, 255, 255)  # White color
                # Draw the text on the image
                draws[idx].text(text_position, image_order[idx], fill="red", font=font)

            prediction_gdino = predictions_gdino[idx]
            bboxes_list_img = []
            for box, score, labels in zip(prediction_gdino["boxes"], prediction_gdino["scores"], prediction_gdino["labels"]):
                if 'ego' in labels:
                    continue
                box = [round(x, 2) for x in box.tolist()]
                center_box=[int((box[0]+box[2])/2), int((box[1]+box[3])/2)]
                if args.json_cogmap:
                    cog_map["detections"].append({"view":image_order[idx],"object":labels,"3D location":world_coords[idx,center_box[1],center_box[0]].round()})
                elif args.visual_cogmap:
                    center_coord=world_coords[idx,center_box[1],center_box[0]]
                    center_coord=[center_coord[0].item(),center_coord[2].item()]
                    objects_xyz.append(center_coord)
                    objects_names.append(labels)
                else:
                    cog_map+=f"\n{image_order[idx]}: detected {labels} at 3D location {world_coords[idx,center_box[1],center_box[0]].round()}"
                bboxes_list_img.append(box)
                if debug:
                    draws[idx].rectangle(box, outline="red", width=3)
                    center_coord=world_coords[idx,center_box[1],center_box[0]]
                    center_coord=[center_coord[0].item(),center_coord[2].item()]
                    objects.append(center_coord)
                    legends.append(labels)


                    font = ImageFont.truetype("assets/Arial.ttf", 25)  # Adjust font size (20) here
                    pixel_width = box[2] - box[0]
                    pixel_height = box[3] - box[1]
                    Z=prediction_depth[idx,center_box[1],center_box[0]]
                    width = (pixel_width*Z)/cams_K[idx, 0, 0]
                    height = (pixel_height*Z)/cams_K[idx, 1, 1]
                    width=int(width*100)/100
                    height=int(height*100)/100
    
                    text = f"{labels}|height:{height}|width:{width}|3D:{torch.round(world_coords[idx,center_box[1],center_box[0]]).tolist()}" 

                    text_position = (box[0], box[1] - 25)  # Adjust vertical offset

                    draws[idx].text(text_position, text, fill="red",font=font)

            bboxes_list.append(bboxes_list_img)
            if debug:
                # Save or display
                images[idx].show()
                images[idx].save(image_names[idx])

        if args.wo_vision:
            input_text = f"\n{str(cog_map)}{question}"
            response = model.chat(tokenizer, None, input_text, generation_config,)
        elif args.wo_vision_llm:
            input_text = f"\n{str(cog_map)}{question}"
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]            
        else:
            if args.visual_cogmap:
                question = insert_cogmap_to_internvl_input(question,image_path_sorted,cogmap='\nVisual Cognitive Map of the Scene: <image>')
                path = plot_cog_map(objects_xyz, objects_names, 'logs/cogmap.png')
                images.append(Image.open(path))
                pixel_values, num_patches_list = prepare_images_internvl(images)
            else:
                question = insert_cogmap_to_internvl_input(question,image_path_sorted,str(cog_map))
                pixel_values, num_patches_list = prepare_images_internvl(images)


            response, history = model.chat(
                tokenizer, pixel_values, question, generation_config,
                num_patches_list=num_patches_list,
                history=None, return_history=True
            )

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
        save_path[category] = f"logs/{model_name}-ego3dvlm/{category}.jsonl"
        
        ## We have two categories  with exact number as output and the rest if categories are multi-choice
        if category in ['Ego_Centric_Absolute_Distance','Object_Centric_Absolute_Distance']:
            multi_choice=False
        else:
            multi_choice=True
        eval_logs(save_path[category],multi_choice=multi_choice)

    
