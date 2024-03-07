experiment_name=llama2_7B_test

log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch ./src/evaluate/eval_yi.py \
--model_path=/your_data_path/llama2-7b \
--input_path=./data/Llama2-7B/test.json \
--output_path=./result/Llama2-7B/model_ans.jsonl \
--score_path=./result/Llama2-7B/score.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &
