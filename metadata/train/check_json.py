import json

def check_json_items(json_data):
    final_data = []
    for index, item in enumerate(json_data):
        flag = True
        for i, value in enumerate(item):
            if not isinstance(value, str):
                if 'question' in value:
                    item[0] = value['question']
                else:
                    flag = False
        if flag:
            final_data.append(item)
    return final_data

# 读取 JSON 文件
json_file_path = '/mntcephfs/data/med/xidong/Medbase/metadata/train/Medical_Wiki_en/Medical_Wiki_en1.json'  # 替换为你的 JSON 文件路径
with open(json_file_path, 'r') as json_file:
    try:
        json_data = json.load(json_file)
        final_data = check_json_items(json_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

with open('final.json', 'w') as file:
    file.write(json.dumps(final_data, ensure_ascii=False, indent=2))
