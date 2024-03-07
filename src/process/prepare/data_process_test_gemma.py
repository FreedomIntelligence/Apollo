import argparse
import json
import os
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

question_prompt_en_choice_shot = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{options}
Assistant:The correct answer is {answer}.<eos>
"""
question_prompt_en_choice = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{options}
Assistant:"""

question_prompt_en_pubmed_shot = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to C. Choose ‘yes’ or ‘no’ if the evidence in the context supports a definitive answer. Choose ‘maybe’ if the evidence in the context does not support a definitive answer.
Context: {context} 
Question: {question}
Options:
{options}
Assistant:The correct answer is {answer}.<eos>
"""

question_prompt_en_pubmed = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to C. Choose ‘yes’ or ‘no’ if the evidence in the context supports a definitive answer. Choose ‘maybe’ if the evidence in the context does not support a definitive answer.
Context: {context} 
Question: {question}
Options:
{options}
Assistant:"""


question_prompt_zh_choice_shot = """User:您是一名医生，正在回答现实世界的医学考试问题。请从A到D中选择一个正确答案。
问题: {question}
选项:
{options}
Assistant:正确答案是{answer}.<eos>
"""
question_prompt_zh_choice = """User:您是一名医生，正在回答现实世界的医学考试问题。请从A到D中选择一个正确答案。
问题: {question}
选项:
{options}
Assistant:"""


question_prompt_es_choice_shot = """User:Usted es un médico que responde preguntas de exámenes médicos del mundo real. Elija una respuesta correcta de la A a la D.
pregunta: {question}
Opciones:
{options}
Assistant:La respuesta correcta es {answer}.<eos>
"""
question_prompt_es_choice = """User:Usted es un médico que responde preguntas de exámenes médicos del mundo real. Elija una respuesta correcta de la A a la D.
pregunta: {question}
Opciones:
{options}
Assistant:"""

question_prompt_fr_choice_shot = """User:Vous êtes un médecin et répondez à des questions d'examen médical du monde réel. Veuillez choisir une bonne réponse de A à E.
question: {question}
Possibilités:
{options}
Assistant:La bonne réponse est {answer}.<eos>
"""
question_prompt_fr_choice = """User:Vous êtes un médecin et répondez à des questions d'examen médical du monde réel. Veuillez choisir une bonne réponse de A à E.
question: {question}
Possibilités:
{options}
Assistant:"""

question_prompt_hi_choice_shot = """User:आप एक डॉक्टर हैं जो वास्तविक दुनिया की मेडिकल परीक्षा के सवालों का जवाब दे रहे हैं। कृपया A से D तक सही उत्तर चुनें।
सवाल: {question}
विकल्प:
{options}
Assistant:सही उत्तर है{answer}.<eos>
"""
question_prompt_hi_choice = """User:आप एक डॉक्टर हैं जो वास्तविक दुनिया की मेडिकल परीक्षा के सवालों का जवाब दे रहे हैं। कृपया A से D तक सही उत्तर चुनें।
सवाल: {question}
विकल्प:
{options}
Assistant:"""

question_prompt_ar_choice_shot = """User:أنت طبيب يجيب على أسئلة الفحص الطبي في العالم الحقيقي. الرجاء اختيار الإجابة الصحيحة من أ إلى د.
سؤال: {question}
خيارات:
{options}
Assistant:{answer}الإجابة الصحيحة هي.<eos>
"""
question_prompt_ar_choice = """User:أنت طبيب يجيب على أسئلة الفحص الطبي في العالم الحقيقي. الرجاء اختيار الإجابة الصحيحة من أ إلى د.
سؤال: {question}
خيارات:
{options}
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
        elif source in ['headqa']:
            few_shot_prompt = question_prompt_es_choice_shot
            question_prompt = question_prompt_es_choice
        elif source in ['frenchmedmcqa']:
            few_shot_prompt = question_prompt_fr_choice_shot
            question_prompt = question_prompt_fr_choice
        elif source in ['mmlu-medical-ar']:
            few_shot_prompt = question_prompt_ar_choice_shot
            question_prompt = question_prompt_ar_choice
        elif source in ['mmlu-medical-hi']:
            few_shot_prompt = question_prompt_hi_choice_shot
            question_prompt = question_prompt_hi_choice
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
            real_question = question_prompt.format(**item)
            real_question_len = len(real_question)
            for sample in random_samples:
                sample = few_shot_prompt.format(**sample)
                if len(question) + real_question_len + len(sample) < 4096:
                    question += sample
            question += real_question
            if len(question)>4096:
                continue
            if debug:
                print(question)
                debug=False
            
            tmp_dict['source_question'] = item['question']
            tmp_dict['source_option'] = item['options']
            tmp_dict['question'] = question
            tmp_dict['answer'] = item['answer'][1]
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