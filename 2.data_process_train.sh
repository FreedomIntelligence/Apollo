# need change 4 place
# Please set the wandb key in the python file (e.g ./src/process/prepare/data_process_train_gemma.py)

mkdir wandb_logs

experiment_name=Gemma_MixTrain_Data
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log


python ./src/process/prepare/data_process_train_gemma.py \
--data_path ./metadata/train/mixTrain.json \
--model_path /your/path/to/gemma-2b \
--wandb_log ./wandb_logs \
--experiment_name ${experiment_name} \
--save_path ./data/Gemma/mixTrain > ${log_folder}/$log_name 2>&1 &