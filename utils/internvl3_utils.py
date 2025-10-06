import torch
from PIL import Image,ImageDraw, ImageFont,ImageOps
from transformers import  AutoConfig
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import ImageDraw

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map




def load_image(image_file, bboxes=None,bboxes_cap=None, input_size=448, max_num=12):

    if isinstance(image_file,Image.Image):
        image = image_file.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    # Optional: draw bboxes on the original image before dynamic_preprocess
    if bboxes:
        draw = ImageDraw.Draw(image)
        for bbox,bbox_cap in zip(bboxes,bboxes_cap):
            draw.rectangle(bbox, outline='red', width=3)
            if bbox_cap:
                text_position = (bbox[0], bbox[1] - 25)  
                font = ImageFont.truetype("assets/Arial.ttf", 25) 
                draw.text(text_position, bbox_cap, fill="red",font=font)

    transform = build_transform(input_size=input_size)

    # Apply dynamic preprocessing (e.g., cropping/patching)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)

    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values

def prepare_images_internvl(image_paths, bboxes_list=None, bboxes_list_cap=None):
    pixel_vals = ()

    if bboxes_list is None:
        bboxes_list = [None] * len(image_paths)

    if bboxes_list_cap is None:
        bboxes_list_cap = [None] * len(image_paths)

    for img_path, bboxes, bboxes_cap in zip(image_paths, bboxes_list, bboxes_list_cap):
        pixel_vals += (load_image(img_path, bboxes=bboxes, bboxes_cap=bboxes_cap, max_num=1).to(torch.bfloat16).cuda(),)

    pixel_values = torch.cat(pixel_vals, dim=0)
    num_patches_list = [pixel_vals[i].size(0) for i in range(len(pixel_vals))]

    return pixel_values, num_patches_list

