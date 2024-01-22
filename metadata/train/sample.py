import os
import json
import random
import shutil

def sample_items(input_file, output_folder, sample_size=100):
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机采样指定数量的项
    sampled_items = random.sample(data, min(sample_size, len(data)))

    # 获取原文件名（去除路径和扩展名）
    file_name = os.path.splitext(os.path.basename(input_file))[0]

    # 构建目标文件路径
    output_file = os.path.join(output_folder, f"{file_name}_sampled.json")

    # 写入采样后的数据到目标文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_items, f, ensure_ascii=False, indent=2)

    print(f"采样结果已保存到: {output_file}")

# 指定输入文件夹和输出文件夹
input_folder = "./data_directory_medbase"
output_folder = "./data_directory_medbase_sample"

# 遍历输入文件夹下的所有JSON文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".json"):
        input_file = os.path.join(input_folder, file_name)
        sample_items(input_file, output_folder)
