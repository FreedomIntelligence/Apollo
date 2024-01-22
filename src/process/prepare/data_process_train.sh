experiment_name=qwen1_8B_v1_datapre
log_folder="/223040239/medbase/logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log


python /223040239/medbase/src/process/prepare/data_process_train_qwen_v1.py \
--data_path /223040239/medbase/metadata/train/medbase_train.json \
--model_path /sds_wangby/models/Qwen-1_8B \
--wandb_log /223040239/medbase/wandb_logs \
--experiment_name ${experiment_name} \
--save_path /223040239/medbase/data/Qwen-1.8B/Medbase_v1_train > ${log_folder}/$log_name 2>&1 &


