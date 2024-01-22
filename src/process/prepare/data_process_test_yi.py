import argparse
import json
import os
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

question_prompt_en_choice_shot = """<|User|>:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{option}
Assistant:The correct answer is {anwer}.<|endoftext|>
"""
question_prompt_en_choice = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{option}
Assistant:"""

question_prompt_en_pubmed_shot = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to C. Choose ‘yes’ or ‘no’ if the evidence in the context supports a definitive answer. Choose ‘maybe’ if the evidence in the context does not support a definitive answer.
Context: {context} 
Question: {question}
Options:
{option}
Assistant:The correct answer is {anwer}.<|endoftext|>
"""

question_prompt_en_pubmed = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to C. Choose ‘yes’ or ‘no’ if the evidence in the context supports a definitive answer. Choose ‘maybe’ if the evidence in the context does not support a definitive answer.
Context: {context} 
Question: {question}
Options:
{option}
Assistant:"""


question_prompt_zh_choice_shot = """User:您是一名医生，正在回答现实世界的医学考试问题。请从A到D中选择一个正确答案。
问题: {question}
选项:
{option}
Assistant:正确答案是{anwer}.<|endoftext|>
"""
question_prompt_zh_choice = """User:您是一名医生，正在回答现实世界的医学考试问题。请从A到D中选择一个正确答案。
问题: {question}
选项:
{option}
Assistant:"""


    
def preprocess(args):
    data_final = []
    with open(args.data_path, 'r') as file:
        data = json.load(file)
    grouped_items = {}
    for item in data:
        source = item.get("source")
        if source not in grouped_items:
            grouped_items[source] = []
        grouped_items[source].append(item)

    for source, items in grouped_items.items():
        debug = True 
        print(f'*********************{source}****************************')
        if source in ['cmb-single', 'cmexam', 'cmmlu-medical', 'medqa-mcmle']:
            few_shot_prompt = question_prompt_zh_choice_shot
            question_prompt = question_prompt_en_choice
        elif source in ['medmcqa', 'medqa-usmle', 'mmlu-medical']:
            few_shot_prompt = question_prompt_en_choice_shot
            question_prompt = question_prompt_en_choice
        else:
            few_shot_prompt = question_prompt_en_pubmed_shot
            question_prompt = question_prompt_en_pubmed
            
        for item in items:
            random_samples = random.sample(items, args.few_shot+1)
            question = ''
            tmp_dict = {}
            # in case item in random_samples
            if item in random_samples:
                random_samples.remove(item)
            else:
                random_samples = random_samples[:-1]
            for sample in random_samples:
                question += few_shot_prompt.format(**sample)
            question += question_prompt.format(**item)
            if debug:
                print(question)
                debug=False
            
            tmp_dict['source_question'] = item['question']
            tmp_dict['source_option'] = item['option']
            tmp_dict['question'] = question
            tmp_dict['answer'] = item['anwer'][1]
            tmp_dict['source'] = item['source']
            data_final.append(tmp_dict)
                
    with open(args.save_path, 'w', encoding='utf-8') as file:
        json.dump(data_final, file, ensure_ascii=False, indent=2)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of Data Preprocess')

    # Model Args
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--few_shot', default='', type=int)
    args = parser.parse_args()

    preprocess(args)  