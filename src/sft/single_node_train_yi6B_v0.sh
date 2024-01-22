#!/bin/bash
#python *.py

cd /223040239/medbase/src/sft
process_port=29503
experiment_name=Yi6B_v0_train
model_dir=/sds_wangby/models/Yi-6B
# ckpt_dir=/mntcephfs/data/med/xidong/Medbase/ckpts/qwen-1_8b_v1/checkpoint-0-2228
train_data_file=/223040239/medbase/data/Yi-6B/Medbase_v0_train
dev_data_file=/223040239/medbase/data/Yi-6B/dev.json
output_dir=/223040239/medbase/ckpts
log_folder="/223040239/medbase/logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch \
    --config_file /223040239/medbase/src/sft/training_config/zero.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port ${process_port} \
    --num_cpu_threads_per_process 8 \
    --deepspeed_multinode_launcher standard /223040239/medbase/src/sft/train_yi_resume_val.py \
    --model_path ${model_dir} \
    --experiment_name ${experiment_name} \
    --gradient_accumulation_steps 4 \
    --train_data_dir ${train_data_file} \
    --dev_data_dir ${dev_data_file} \
    --output_dir ${output_dir} \
    --log_dir /223040239/medbase/wandb_logs \
    --n_epochs 1 \
    --train_bsz_per_gpu 2 \
    --eval_bsz_per_gpu 4 \
    --learning_rate 1e-4 \
    --eval_step -1 \
    --save_step -1 \
    --warmup_rates 0.03 \
    --max_ckpts 3 \
    --gradient_checkpointing  > ${log_folder}/$log_name 2>&1 &

