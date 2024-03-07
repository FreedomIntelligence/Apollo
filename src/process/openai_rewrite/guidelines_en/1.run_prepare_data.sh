#!/bin/bash

python 1.prepare_data.py --input_path ./data/1.dev.jsonl --output_path ./data/2.dev_prepared.jsonl
python 1.2.prepare_data.py --input_path ./data/3.dev_aftgpt.jsonl --output_path ./data/2.dev_prepared.jsonl