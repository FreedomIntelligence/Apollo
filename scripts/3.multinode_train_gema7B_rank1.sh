#!/bin/bash
#python *.py

node_rank=1
master_ip=yourip


cd ./src/sft
process_port=29503
experiment_name=Gemma7B_MixTrain
model_dir=/your/path/to/Gemma-7b
# ckpt_dir=./ckpts/
train_data_file=./data/gemma/MixTrain
dev_data_file=./data/gemma/dev.json
output_dir=./ckpts
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log



CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --config_file ./src/sft/training_config/zero_multi.yaml \
    --num_processes 24 \
    --num_machines 3 \
    --machine_rank ${node_rank} \
    --main_process_ip "${master_ip}" \
    --main_process_port ${process_port} \
    --num_cpu_threads_per_process 8 \
    --deepspeed_multinode_launcher standard ./src/sft/train_gemma_resume_val.py \
    --model_path ${model_dir} \
    --experiment_name ${experiment_name} \
    --gradient_accumulation_steps 8 \
    --train_data_dir ${train_data_file} \
    --dev_data_dir ${dev_data_file} \
    --output_dir ${output_dir} \
    --log_dir ./wandb_logs \
    --n_epochs 1 \
    --train_bsz_per_gpu 2 \
    --eval_bsz_per_gpu 2 \
    --learning_rate 1e-5 \
    --eval_step -1 \
    --save_step -1 \
    --warmup_rates 0.03 \
    --max_ckpts 3 \
    --gradient_checkpointing  > ${log_folder}/rank${node_rank}.log 2>&1 &