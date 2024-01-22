#!/bin/bash

# Set the path to the Python script
python_script="../OpenAIGPT_datagen_multithread.py"

# set parameter
keys_path="../gpt3keys.txt"
input_path="./data/2.rlhf_prepared.jsonl"
output_path="./data/3.rlhf_aftgpt.jsonl"
max_workers=50

python "$python_script" --keys_path "$keys_path" --input_path "$input_path" --output_path "$output_path" --max_workers $max_workers
python ../OpenAIGPT_datagen_multithread.py --keys_path ../gpt4key.txt --input_path ./data/2.patients_en_prepared.jsonl  --output_path ./data/3.patients_en_aftgpt.jsonl --max_workers 300
python ../OpenAIGPT_datagen_multithread.py --keys_path ../gpt4key.txt --input_path ./data/2.guideline_en_aftgpt_prepared.jsonl  --output_path ./data/3.guideline_en_aftgpt_prepared_aftgpt.jsonl --max_workers 300