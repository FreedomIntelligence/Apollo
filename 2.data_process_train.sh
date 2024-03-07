model_name = Qwen
model_path = /your_model_path/Qwen1.5-0.5B
experiment_name=qwenallsftcom_data
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log


python ./src/process/prepare/data_process_train_qwen.py \
--data_path ./metadata/train/sft.json \
--model_path ${model_path} \
--wandb_log ./wandb_logs \
--experiment_name ${experiment_name} \
--save_path ./data/${model_name}/allsftcom > ${log_folder}/$log_name 2>&1 &

