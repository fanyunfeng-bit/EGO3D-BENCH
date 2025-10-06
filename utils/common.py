import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import random
import cv2
import numpy as np
import torch

plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono"]


def plot_cog_map(objects_xyz,objects_names,path):
    # Use a whitegrid with light blue background
    sns.set_theme(style="whitegrid", context="talk")
    fig,ax=plt.subplots(figsize=((5,5)))
    matplotlib.rcParams.update({'font.size': 11})
    x=[point[0] for point in objects_xyz]
    y=[point[1] for point in objects_xyz]
    # plt.xlim(-20,20) 
    # plt.ylim(-20,20) 
    ego_car_x=[-1,-1,1,1,-1]
    ego_car_y=[2.5,-2.5,-2.5,2.5,2.5]
    plt.plot(ego_car_x,ego_car_y,'r',label='ego car',linewidth=5)
    for pt_x,pt_y,name in zip(x,y,objects_names):
        plt.plot(pt_x,pt_y,'s',label=name,markersize=10) 
    plt.ylabel('Z Axis(Front(+) and Back(-))',fontsize=14)
    plt.xlabel('X Axis(Right(+) and Left(-))',fontsize=14)
    plt.axis('equal')
    plt.title('Cog Map',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(np.arange(min(x)-2,max(x)+2, 1)) 
    # plt.yticks(np.arange(min(y)-2,max(y)+2, 1)) 
    plt.grid()
    fig.legend(loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),fontsize=14)

    fig.tight_layout()
    
    fig.savefig(path,bbox_inches='tight')
    # image = Image.open('results/depth_and_gdino/cog_map.png')
    return path


def add_occlusion_by_percentage(image, occlusion_percent=50, max_patch_size=(100, 100)):
    """
    Adds rectangular occlusion patches to an image to cover approximately
    `occlusion_percent` of the total image area.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    total_area = width * height
    occluded_area = 0
    target_occlusion_area = total_area * (occlusion_percent / 100.0)

    while occluded_area < target_occlusion_area:
        # Random patch size
        w = random.randint(20, max_patch_size[0])
        h = random.randint(20, max_patch_size[1])

        # Random top-left corner
        x0 = random.randint(0, width - w)
        y0 = random.randint(0, height - h)

        # Draw rectangle
        color = random.choice([(0, 0, 0), (30, 30, 30), (60, 60, 60)])  # dark occlusion
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)

        # Accumulate occluded area
        occluded_area += w * h

    return image


def apply_motion_blur(pil_image, degree=15, angle=0):
    """
    Applies motion blur to a PIL image.

    Args:
        pil_image: Input PIL image.
        degree: Length of the blur streak.
        angle: Angle of motion in degrees (0 = horizontal, 90 = vertical).
    """
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Create the motion blur kernel
    kernel = np.zeros((degree, degree), dtype=np.float32)
    kernel[int((degree - 1) / 2), :] = np.ones(degree, dtype=np.float32)
    kernel = kernel / degree

    # Rotate the kernel to match the motion angle
    center = (degree / 2, degree / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (degree, degree))

    # Apply the kernel to the image
    blurred = cv2.filter2D(img, -1, kernel)

    # Convert back to PIL format
    return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))




def unproject(intrinsics, poses, depths):
    """
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4)
        depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) #/ 1000       # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords



def strip_question(sample):
    #### This fincton extract the question from the input prompt of Ego3D-Bench.
    #### Note: To apply Ego3D-VLM to other tasks it's better to apply REC model to the question without extra prompt info.
    import re
    question_text = sample['question']
    parts = re.split(r"<image>", question_text)
    return parts[-1]

def insert_cogmap_to_internvl_input(question,imag_paths,cogmap):
    """
    Convert a dataset sample with <image> placeholders into Qwen2.5-VL input format.
    """
    import re

    # Split the text by <image> placeholders
    parts = re.split(r"<image>", question)

    # Build the content list
    content = ""
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            content += part 
        if i < len(imag_paths):  # Add image after each <image> except the last split
            content += "<image>\n"
        if i==len(imag_paths)-1: 
            content += cogmap +'\n\n'
    return content


def convert_to_qwen_input(question,image_paths,cog_map=None):
    """
    Convert a dataset sample with <image> placeholders into Qwen2.5-VL input format.
    """
    import re

    # Split the text by <image> placeholders
    parts = re.split(r"<image>", question)

    # Build the content list
    content = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            content.append({"type": "text", "text": part})
        if i < len(image_paths):  # Add image after each <image> except the last split
            img_path = image_paths[i]
            if img_path:  # skip empty paths
                content.append({"type": "image", "image": img_path})

        if i == len(image_paths)-1 and cog_map:
            content.append({"type": "text", "text": cog_map})

    # Wrap into message format
    messages = [{"role": "user", "content": content}]
    return messages



def pad_images(images):
    # Find max width and height
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Pad each image
    padded_images = []
    for idx,img in enumerate(images):
        delta_w = max_width - img.width
        delta_h = max_height - img.height
        # Compute padding: (left, top, right, bottom)
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2)
        )

        padded_img = ImageOps.expand(img, padding, fill=0)  # fill=0 for black
        padded_images.append(padded_img)
    
    return padded_images
