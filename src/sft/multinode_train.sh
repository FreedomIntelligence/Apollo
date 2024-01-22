#!/bin/bash

# Check if MASTER_ADDR is 'localhost' and update it to the IP address
[ "${MASTER_ADDR}" == "localhost" ] && export MASTER_ADDR=$(hostname -i)


master_ip=$MASTER_ADDR
node_rank=$RANK
nodes=$WORLD_SIZE

# Logging the configuration
cat << EOF
master_ip: $MASTER_ADDR
node_rank: $RANK
EOF

process_port=29501
experiment_name=yi-34b_huatuo2_multi_fsdp
model_dir=/sds_wangby/models/cjy/model_download/Yi-34B
data_file=/sds_wangby/models/cjy/huatuo2/data/huatuov2_pretrain_v2_4_x_Yi-34B_4096_dataset_final
output_dir=/sds_wangby/models/cjy/huatuo2/ckpts
log_folder="/sds_wangby/models/cjy/huatuo2/logs/${experiment_name}"
mkdir -p $log_folder

cd /sds_wangby/models/cjy/huatuo2
# Build the command
cmd="accelerate launch \
		--config_file /sds_wangby/models/cjy/huatuo2/configs/zero2.yaml \
		--num_processes 40 \
		--num_machines 5 \
		--machine_rank ${node_rank} \
		--main_process_ip "${master_ip}" \
		--main_process_port ${process_port} \
		--num_cpu_threads_per_process 8 \
		--deepspeed_multinode_launcher standard /sds_wangby/models/cjy/huatuo2/finetune_wxd_zero2.py \
		--model_path ${model_dir} \
		--experiment_name ${experiment_name} \
		--gradient_accumulation_steps 2 \
		--data_dir  ${data_file} \
		--output_dir ${output_dir} \
		--log_dir /sds_wangby/models/cjy/huatuo2/logs_wandb \
		--n_epochs 2 \
		--train_bsz_per_gpu 1 \
		--eval_bsz_per_gpu 2 \
		--learning_rate 1e-4 \
		--eval_step -1 \
		--save_step -1 \
		--max_seq_len 4096 \
		--warmup_rates 0.03 \
		--max_ckpts 3 \
		--gradient_checkpointing  > ${log_folder}/rank${node_rank}.log 2>&1
"""



# Execute the command
echo "Executing command: $cmd"
eval "$cmd"
