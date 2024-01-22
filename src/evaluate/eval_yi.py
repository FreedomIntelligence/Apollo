
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


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,tokenizer):
        self.data = []
        with open(data_path) as f:
            self.data = json.load(f)
        dist_flag_0 = True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0) else False
        if dist_flag_0:
            print(f'load {len(self.data)} data from {data_path}')
        self.tokenizer = tokenizer
        self.debug = True

    def __getitem__(self, index):
        item = self.data[index]
        return {
            'data': item,
            'input': item['question']
        }

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        batch_query = [x['input'] for x in batch]
        batch_data = [x['data'] for x in batch]
        out_batch = {}
        out_batch['data'] = batch_data
        out_batch['input_ids'] = self.tokenizer(batch_query, return_tensors='pt', padding=True)['input_ids']
        dist_flag_0 = True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0) else False
        if self.debug and dist_flag_0:
            decoded_texts = self.tokenizer.batch_decode(out_batch['input_ids'], skip_special_tokens=False)
            for idx, sample in enumerate(decoded_texts):
                print(f'*******************batch_texts[{idx}]**********************************')
                print(sample)
            self.debug = False
        return out_batch
        

def get_response(batch_input_ids, batch_output_ids, tokenizer, num_return):
    responses_list=[]
    batch_return=[]
    input_len = len(batch_input_ids[0])
    for idx, output_ids in enumerate(batch_output_ids):
        generated_ids = output_ids[input_len:]
        batch_return.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
        if idx % num_return == num_return-1:
            responses_list.append(batch_return)
            batch_return=[]
    return responses_list

def extract_and_choose_answer(pattern, model_answer):
    matches = re.findall(pattern, model_answer)
    option_count = {}
    for match in matches:
        option_count[match.upper()] = option_count.get(match.upper(), 0) + 1

    if not option_count:
        loose_pattern = r'[A-F]'
        if pattern == loose_pattern:
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
        for answer in item["model_answer"]:
            all[source] += 1  
            pattern = r'[（\(]([A-Fa-f])[）\)]'
            extract_answer = extract_and_choose_answer(pattern, answer)
            if debug:
                debug = False
                print(f'extract_answer:{extract_answer}')
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
    

def generate_response(args):
    accelerator = Accelerator()
    
    model_path = args.model_path
    accelerator.print(f'****************model_path:{model_path}******************')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True).half()
    gen_kwargs = {'num_return_sequences': args.num_return, 'max_new_tokens': 16, 'min_new_tokens':2, 'do_sample':False}
    
    
    dataset = TestDataset(args.input_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)

    model = model.eval()
    if dist.is_initialized():
        accelerator.print(f'****************dist.get_world_size():{dist.get_world_size()}******************')

    model, dataloader = accelerator.prepare(model, dataloader)
    accelerator.print(f'******************load_model from {model_path}******************')

    if accelerator.is_main_process:
        fp = open(args.output_path,'w')
        
    dataloader_iterator = tqdm(dataloader, total=len(dataloader)) if accelerator.is_main_process else dataloader
    for batch in dataloader_iterator:
        batch_input_ids = batch["input_ids"]
        batch_data = batch["data"]
        batch_output_ids = accelerator.unwrap_model(model).generate(batch_input_ids, **gen_kwargs)
        batch_responses = get_response(batch_input_ids, batch_output_ids, tokenizer, args.num_return)
        
        
        if dist.is_initialized():
            all_batch_data =  [None] * dist.get_world_size()
            all_batch_responses =  [None] * dist.get_world_size()
            dist.all_gather_object(all_batch_responses, batch_responses)
            dist.all_gather_object(all_batch_data, batch_data)
        else:
            all_batch_data = [batch_data, ]
            all_batch_responses = [batch_responses, ]
                
        all_data = [item for sublist in all_batch_data for item in sublist]
        all_response = [item for sublist in all_batch_responses for item in sublist]
            
        for data, responses in zip(all_data, all_response):
            answer_list = []
            for response in responses:
                answer_list.append(response)
            data['model_answer'] = answer_list
            if accelerator.is_main_process:
                fp.write(json.dumps(data, ensure_ascii=False) +'\n')
                fp.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument("--input_path", type=str, help="path to the input data")
    parser.add_argument("--output_path", type=str, help="path to the output data")
    parser.add_argument("--score_path", type=str, help="path to the score")
    parser.add_argument("--num_return", type=int, help="number of return sequences")
    parser.add_argument("--batch_size", type=int, help="batch size")
    args = parser.parse_args()
    generate_response(args)
    generate_score(args.output_path, args.score_path)

'''

accelerate launch /mntcephfs/data/med/xidong/Medbase/src/evaluate/eval_qwen.py \
--model_path=/mntcephfs/data/med/xidong/checkpoints/Qwen-1_8B \
--input_path=/mntcephfs/data/med/xidong/Medbase/data/Qwen-1.8B/test.json \
--output_path=/mntcephfs/data/med/xidong/Medbase/result/Qwen-1.8B/model_ans.jsonl \
--score_path=/mntcephfs/data/med/xidong/Medbase/result/Qwen-1.8B/score.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &

'''
