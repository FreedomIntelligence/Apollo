import os
import json
import random
from pathlib import Path

# 源目录和目标目录
source_directory = "/mntcephfs/data/med/xidong/Medbase/data/test/zh"
target_directory = "/mntcephfs/data/med/xidong/Medbase/data/dev/exam/zh"

# 获取源目录中的所有JSONL文件
jsonl_files = [file for file in os.listdir(source_directory) if file.endswith(".jsonl")]

# 遍历每个JSONL文件
for jsonl_file in jsonl_files:
    # 构建完整的文件路径
    file_path = os.path.join(source_directory, jsonl_file)

    # 读取JSONL文件中的行
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 计算5%的样本数量
    sample_size = int(len(lines) * 0.05)

    # 随机抽取5%的样本
    sampled_lines = random.sample(lines, sample_size)

    # 构建目标文件路径
    target_file_path = os.path.join(target_directory, jsonl_file)

    # 写入抽取的样本到目标文件
    with open(target_file_path, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

print("抽取并存储完成。")
