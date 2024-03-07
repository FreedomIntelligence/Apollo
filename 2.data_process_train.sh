experiment_name=yiallsftcom_Data
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log


python ./src/process/prepare/data_process_train_yi.py \
--data_path ./metadata/train/sft.json \
--model_path /your_model_path/Yi-34B \
--wandb_log ./wandb_logs \
--experiment_name ${experiment_name} \
--save_path ./data/Yi/allsftcom > ${log_folder}/$log_name 2>&1 &

