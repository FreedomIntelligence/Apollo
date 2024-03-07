#!/bin/bash
#python *.py

process_port=29502
experiment_name=qwen0_5_allsft
model_dir=/your_model_path/Qwen1.5-0.5B
train_data_file=./data/Qwen/allsftData
dev_data_file=./data/Qwen/dev.json
output_dir=./ckpts
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch \
    --config_file ./src/sft/training_config/zero.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port ${process_port} \
    --num_cpu_threads_per_process 8 \
    --deepspeed_multinode_launcher standard ./src/sft/train_qwen_resume_val.py \
    --model_path ${model_dir} \
    --experiment_name ${experiment_name} \
    --gradient_accumulation_steps 8 \
    --train_data_dir ${train_data_file} \
    --dev_data_dir ${dev_data_file} \
    --output_dir ${output_dir} \
    --log_dir ./wandb_logs \
    --n_epochs 1 \
    --train_bsz_per_gpu 4 \
    --eval_bsz_per_gpu 4 \
    --learning_rate 1e-4 \
    --eval_step -1 \
    --save_step -1 \
    --warmup_rates 0.03 \
    --max_ckpts 5 \
    --gradient_checkpointing  > ${log_folder}/$log_name 2>&1 &
