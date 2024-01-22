experiment_name=llama2_7B_test

log_folder="/223040239/medbase/logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch /223040239/medbase/src/evaluate/eval_yi.py \
--model_path=/sds_wangby/models/llama2-7b \
--input_path=/223040239/medbase/data/Llama2-7B/test.json \
--output_path=/223040239/medbase/result/Llama2-7B/model_ans.jsonl \
--score_path=/223040239/medbase/result/Llama2-7B/score.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &
