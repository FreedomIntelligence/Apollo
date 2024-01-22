import os
import json

def merge_json_files(root_path):
    merged_data = {}
    for file_name in os.listdir(root_path):
        
        if file_name.endswith(".json"):
            file_path = os.path.join(root_path, file_name)
            print(file_path)
            # Read the content of the JSON file
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)

            # Add the data to the merged dictionary using the file name as the key
            merged_data[file_name[:-5]] = json_data

    # Path for the merged file
    merged_file_path = os.path.join(root_path, "medbase_train.json")

    # Write the merged JSON data to a new file
    with open(merged_file_path, "w", encoding="utf-8") as merged_file:
        json.dump(merged_data, merged_file, ensure_ascii=False, indent=2)

    print(f"Merged file has been saved as: {merged_file_path}")

# Replace with the actual directory path
merge_json_files("/223040239/medbase/metadata/train/data_directory_medbase")
