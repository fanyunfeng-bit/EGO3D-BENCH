import json
import re
from sklearn.metrics import mean_squared_error
import numpy as np

def extract_number_from_answer_tag_mult_choice(text):
    # Extract content inside <answer>...</answer>
    # match = re.search(r'<answer>(.*?)</answer>', text)
    start=text.find('<answer>')
    end=text.find('</answer>')
    
    if start!=-1:
        match = text[start+8:end].replace('/n','').strip().lower()
        # print(match)
    else:
        match=text.replace('/n','').strip().lower()
    return match


def extract_number_from_answer_tag_exact_num(text):
    # # Extract content inside <answer>...</answer>
    # match = re.search(r'<answer>(.*?)</answer>', text)
    # start=text.find('<answer>')
    # end=text.find('</answer>')
    # if start!=-1:
    #     match = text[start+8:end]
    # else:
    #     match=text[:end]
    # if text:
    content = text
    # Extract the first number (int or float) from the content
    number_match = re.search(r'[-+]?\d*\.?\d+', content)
    if number_match:
        return float(number_match.group()) if '.' in number_match.group() else int(number_match.group())
    return None


def eval_multi_choice(file_name):
    count=0
    correct=0
    file=open(file_name,'r')
    for line in file:
        count+=1
        data=json.loads(line)
        if extract_number_from_answer_tag_mult_choice(data["Processed_Pred"])==data["GT"].lower():
            correct+=1
    sub_category=file_name.split('/')[-1].split('.')[0]
    print(f'Accuracy of {sub_category} is:{correct/count}')

def eval_exact_num(file_name):
    file=open(file_name,'r')
    y_true,y_pred=[],[]
    for line in file:
        data=json.loads(line)
        pred=extract_number_from_answer_tag_exact_num(data["Processed_Pred"])
        gt=(data["GT"])
        if pred:
            pred=min(float(pred),100)
            gt=float(gt)
            y_true.append(float(gt))
            y_pred.append(float(pred))
    sub_category=file_name.split('/')[-1].split('.')[0]
    print(f'RMSE of {sub_category} is:{np.sqrt(mean_squared_error(y_true,y_pred))}')

def eval_logs(log_file,multi_choice):
    if multi_choice:
        eval_multi_choice(log_file)
    else:
        eval_exact_num(log_file)

