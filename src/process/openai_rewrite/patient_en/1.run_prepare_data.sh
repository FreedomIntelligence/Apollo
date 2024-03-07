#!/bin/bash

# Set the paths for input and output files
input_path="./data/1.rlhf.jsonl"
output_path="./data/2.rlhf_prepared.jsonl"

# run Python scripts
python 1.prepare_data.py --input_path "$input_path" --output_path "$output_path"

python 1.prepare_data.py --input_path ./data/1.dev_en.jsonl --output_path ./data/2.dev_prepared.jsonl
python 1.2.prepare_data.py --input_path ./data/3.dev_aftgpt.jsonl --output_path ./data/2.dev_aftgpt_prepared.jsonl