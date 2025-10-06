def convert_to_qwen_input(question,image_paths,cog_map):
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

        if i == len(image_paths)-1:
            content.append({"type": "text", "text": cog_map})

    # Wrap into message format
    messages = [{"role": "user", "content": content}]
    return messages
def strip_question(sample):
    import re
    question_text = sample['question']
    parts = re.split(r"<image>", question_text)
    return parts[-1]


def insert_cogmap_to_input(question,imag_paths,cogmap):
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

# Example usage
sample = {
    'images': {
        'Front_Left': 'n015-...CAM_FRONT_LEFT__1533201470404874.jpg',
        'Front': 'n015-...CAM_FRONT__1533201470412460.jpg',
        'Front_Right': 'n015-...CAM_FRONT_RIGHT__1533201470420339.jpg',
        'Back_Right': 'n015-...CAM_BACK_RIGHT__1533201470427893.jpg',
        'Back': 'n015-...CAM_BACK__1533201470437525.jpg',
        'Back_Left': 'n015-...CAM_BACK_LEFT__1533201470447423.jpg',
    },
    'question': """These are six camera views mounted on an ego car
Front Left view: <image>
Front view: <image>
Front Right view: <image>
Back Right view: <image>
Back view: <image>
Back Left view: <image>
How far, in meters, is black bmw hatchback stopped at the barrier in the front view from pedestrian wearing red crossing the driveway in the front view?"""
}

qwen_input = convert_to_qwen_input(sample["question"],list(sample['images']),cog_map='This is Cogmap ...')
print(qwen_input)