experiment_name=Yi6B_test

log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch ./src/evaluate/eval_yi.py \
--model_path=/your_data_path/Yi-6B \
--input_path=./data/Yi-6B/dev.json \
--output_path=./result/Yi-6B/model_ans.jsonl \
--score_path=./result/Yi-6B/score.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &
