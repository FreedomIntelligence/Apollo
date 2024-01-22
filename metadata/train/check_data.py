import os
import json
from tqdm import tqdm

def process_json_files(folder_path):
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                print(file_path)
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}")
                    continue

                # 遍历文件中的每个项目
                for item in tqdm(data):
                    # 确保项目是字符串列表
                    if isinstance(item, list):
                        # 遍历列表中的每个字符串
                        for idx, string in enumerate(item):
                            # 检查字符串长度
                            if len(string) < 6:
                                print(f"File: {filename}, Index: {idx}, String: '{string}'")

# 使用示例
folder_path = '/223040239/medbase/metadata/train/General_en'  # 替换为你的文件夹路径
process_json_files(folder_path)
