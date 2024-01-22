import json
import os
from nltk import ngrams
from nltk.metrics import jaccard_distance
from tqdm import tqdm

def read_question_list(file_path):
    with open(file_path, 'r') as f:
        question_list = json.load(f)
    return question_list

def calculate_jaccard_similarity(question, item):
    n = 2  # 设置ngram的大小
    question_ngrams = set(ngrams(question, n))
    item_ngrams = set(ngrams(item, n))
    jaccard_coefficient = 1 - jaccard_distance(question_ngrams, item_ngrams)
    return jaccard_coefficient

def process_json_file(input_path, output_path, question_list):
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_items = []
    if 'zh' in input_path:
            question_list = question_list[8327:]
    else:
        question_list = question_list[:8327]
            
    for item in tqdm(data):
        question = item[0]
        
        # similarity = max(calculate_jaccard_similarity(question, q) for q in question_list)
        similarity = max([question==q for q in question_list])
        if not similarity:
            processed_items.append(item)

    with open(output_path, 'w') as f:
        json.dump(processed_items, f, indent=2)

    return len(data), len(processed_items)

def main(input_folder, output_folder, question_list_path, statistic_output_path):
    question_list = read_question_list(question_list_path)
    statistics = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json') and file_name in ['Medical_Exam_en.json', 'Medical_Exam_zh.json']:
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            print(f"Processing: {input_path}")

            total_items, processed_items = process_json_file(input_path, output_path, question_list)
            statistics.append({
                'file': file_name,
                'total_items_before': total_items,
                'total_items_after': processed_items
            })

    with open(statistic_output_path, 'w') as f:
        json.dump(statistics, f, indent=2)

if __name__ == "__main__":
    input_folder = "./data_directory"
    output_folder = "./data_directory_clean"
    question_list_path = "test_questions.json"
    statistic_output_path = "./data_directory_clean/statistic.json"

    main(input_folder, output_folder, question_list_path, statistic_output_path)
