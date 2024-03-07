import json
import re


def extract_dialogues_from_model_answer(model_answer):
    dialogues = []
    try:
        # Use regular expressions to extract conversations between patients and doctors
        patient_dialogues = re.findall(
            r"<Patient>(.*?)</Patient>", model_answer, re.DOTALL
        )
        doctor_dialogues = re.findall(
            r"<Doctor>(.*?)</Doctor>", model_answer, re.DOTALL
        )

        # Combine conversations into a list
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
            #Read JSONL file line by line
            for line in file:
                json_data = json.loads(line)

                # Check whether the JSON contains the "model_answer" field
                if "model_answer" in json_data:
                    model_answer = json_data["model_answer"]

                    # Extract conversations and add to result list
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
