experiment_name=Qwen_1_8B_test

log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch ./src/evaluate/eval_qwen.py \
--model_path=/your_data_path/Qwen-1.8B \
--input_path=./data/Qwen-1.8B/test.json \
--output_path=./result/Qwen-1.8B/model_ans.jsonl \
--score_path=./result/Qwen-1.8B/score.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &
