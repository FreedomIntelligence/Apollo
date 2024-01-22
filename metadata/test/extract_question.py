import json

def extract_questions_from_json(json_file_path, output_json_path):
    questions_list = []

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for item in data:
            if 'question' in item:
                questions_list.append(item['question'])

    # 将问题列表存储为新的 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(questions_list, output_file, ensure_ascii=False, indent=2)

# 用法示例
input_json_file_path = 'medbase_test.json'
output_json_file_path = 'test_questions.json'
extract_questions_from_json(input_json_file_path, output_json_file_path)
