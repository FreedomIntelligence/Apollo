import json
import re


def extract_dialogues_from_model_answer(model_answer):
    dialogues = []
    try:
        # 使用正则表达式提取病人和医生的对话
        patient_dialogues = re.findall(
            r"<Patient>(.*?)</Patient>", model_answer, re.DOTALL
        )
        doctor_dialogues = re.findall(
            r"<Doctor>(.*?)</Doctor>", model_answer, re.DOTALL
        )

        # 将对话组合成一个list
        dialogues = list(zip(patient_dialogues, doctor_dialogues))
        final_list = []
        for item in dialogues:
            for i in item:
                final_list.append(i)
    except Exception as e:
        print(f"Error extracting dialogues: {str(e)}")

    return final_list


def process_jsonl_file(input_file, output_file):
    result_data = []

    try:
        with open(input_file, "r", encoding="utf-8") as file:
            # 逐行读取JSONL文件
            for line in file:
                json_data = json.loads(line)

                # 检查JSON中是否包含"model_answer"字段
                if "model_answer" in json_data:
                    model_answer = json_data["model_answer"]

                    # 提取对话并添加到结果列表
                    dialogues = extract_dialogues_from_model_answer(model_answer)
                    if not len(dialogues):
                        continue
                    result_data.append(dialogues)

    except Exception as e:
        print(f"Error processing JSONL file: {str(e)}")

    # 将结果保存为JSON文件
    with open(output_file, "w", encoding="utf-8") as output_file:
        json.dump(result_data, output_file, ensure_ascii=False, indent=2)


# 用法示例
input_file_path = "./data/3.patients_en_aftgpt.jsonl"
output_file_path = "./data/4.patients_en_aftgpt.json"

process_jsonl_file(input_file_path, output_file_path)
