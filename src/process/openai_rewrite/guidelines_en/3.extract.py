import json


def extract_and_save(input_jsonl, output_json):
    # 创建一个空列表用于存储提取的数据
    extracted_data = []
    train_data = []
    count = 0
    # 逐行读取JSONL文件并提取指定字段
    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            if len(json_data.get("model_answer", "")) < 32:
                continue

            # 提取指定字段
            extracted_item = {
                "question": json_data.get("model_question", ""),
                "answer": json_data.get("model_answer", ""),
                "reference": json_data.get("reference", ""),
                "id": json_data.get("id", ""),
            }
            list_item = [
                json_data.get("model_question", ""),
                json_data.get("model_answer", ""),
            ]

            # 将提取的数据添加到列表中
            extracted_data.append(extracted_item)
            train_data.append(list_item)
            count += 1

    # 将提取的数据写入新的JSON文件
    print(f"sum_count:{count}")
    with open(output_json, "w", encoding="utf-8") as output:
        json.dump(extracted_data, output, indent=2, ensure_ascii=False)
    with open(train_json, "w", encoding="utf-8") as output:
        json.dump(train_data, output, indent=2, ensure_ascii=False)


# 指定输入JSONL文件和输出JSON文件名
input_jsonl = "./data/3.guideline_en_aftgpt_prepared_aftgpt.jsonl"
output_json = "./data/4.guideline_en_extracted.json"
train_json = "./data/4.guideline_en_train.json"

# 调用提取和保存函数
extract_and_save(input_jsonl, output_json)
