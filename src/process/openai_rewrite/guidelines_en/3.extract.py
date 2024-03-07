import json


def extract_and_save(input_jsonl, output_json):
    # Create an empty list to store the extracted data
    extracted_data = []
    train_data = []
    count = 0
    # Read the JSONL file line by line and extract the specified fields
    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            if len(json_data.get("model_answer", "")) < 32:
                continue

            #Extract specified fields
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

            # Add the extracted data to the list
            extracted_data.append(extracted_item)
            train_data.append(list_item)
            count += 1

    # Write the extracted data into a new JSON file
    print(f"sum_count:{count}")
    with open(output_json, "w", encoding="utf-8") as output:
        json.dump(extracted_data, output, indent=2, ensure_ascii=False)
    with open(train_json, "w", encoding="utf-8") as output:
        json.dump(train_data, output, indent=2, ensure_ascii=False)


#Specify the input JSONL file and output JSON file name
input_jsonl = "./data/3.dev_aftgpt_prepared_aftgpt.jsonl"
output_json = "./data/4.dev_extracted.json"
train_json = "./data/4.dev_train.json"

# Call the extraction and saving functions
extract_and_save(input_jsonl, output_json)
