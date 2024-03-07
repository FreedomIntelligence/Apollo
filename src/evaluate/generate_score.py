
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
import argparse
from accelerate import Accelerator
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
from collections import defaultdict
import json

def extract_and_choose_answer(pattern, model_answer):
    if '\n' in model_answer:
        model_answer_split = model_answer.split('\n')
        for model_answer_i in model_answer_split:
            if len(model_answer_i):
                model_answer = model_answer_i
                break
    
    matches = re.findall(pattern, model_answer)
    option_count = {}
    for match in matches:
        option_count[match.upper()] = option_count.get(match.upper(), 0) + 1

    if not option_count:
        # else use loose pattern
        loose_pattern = r'[A-F]'
        if pattern == loose_pattern:
            if model_answer == 'Yes.':
                return 'A'
            elif model_answer == 'No.':
                return 'B'
            else:
                return None
        else:
            return extract_and_choose_answer(loose_pattern, model_answer)
        
    max_count = max(option_count.values())
    max_options = [option for option, count in option_count.items() if count == max_count]
    return max_options[0]
    


def generate_score(result_path, score_path):
    with open(result_path, 'r', encoding='utf-8') as jsonl_file:
        json_objects = [json.loads(line.strip()) for line in jsonl_file]
        
    all = defaultdict(int)
    right = defaultdict(int)
    accuracy_dict = defaultdict(int)
    
    print(f'****Total:{len(json_objects)}****')
    debug = True
    for item in json_objects:
        source = item["source"]
        answer = item["model_answer"]
        all[source] += 1  
        pattern = r'[（\(]([A-Fa-f])[）\)]'
        extract_answer = extract_and_choose_answer(pattern, answer)
        if debug:
            debug = False
            print(f'extract_answer:{extract_answer}')
            print(answer)
            right_answer = item['answer']
            print(f'right_answer:{right_answer}')
        if item['answer'] == extract_answer:
            right[source] += 1
                
            
    print(f'all:{all}')
    print(f'right:{right}')        
                
    for key in right:
        accuracy_dict[key] = right[key] / all[key]
            
    with open(score_path, "w", encoding="utf8") as f:
        json.dump(accuracy_dict, f, indent=4)
    
    print(f'***********score_result save in {score_path}*************')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", '-a', type=str, help="path to the output data")
    parser.add_argument("--score_path", '-o', type=str, help="path to the score")
    args = parser.parse_args()
    generate_score(args.output_path, args.score_path)
    # merge_result()
