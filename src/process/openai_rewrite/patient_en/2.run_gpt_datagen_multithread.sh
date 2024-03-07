#!/bin/bash

python "$python_script" --keys_path "$keys_path" --input_path "$input_path" --output_path "$output_path" --max_workers $max_workers
python ../OpenAIGPT_datagen_multithread.py --keys_path ../gpt4key.txt --input_path ./data/2.dev_prepared.jsonl  --output_path ./data/3.dev_aftgpt.jsonl --max_workers 300
python ../OpenAIGPT_datagen_multithread.py --keys_path ../gpt4key.txt --input_path ./data/2.dev_aftgpt_prepared.jsonl  --output_path ./data/3.dev_aftgpt_prepared_aftgpt.jsonl --max_workers 300