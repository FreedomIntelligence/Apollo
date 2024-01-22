experiment_name=Yi6B29856_test

log_folder="/223040239/medbase/logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch /223040239/medbase/src/evaluate/eval_yi.py \
--model_path=/223040239/medbase/ckpts/Yi6B_v0_train/checkpoint-0-29856/tfmr \
--input_path=/223040239/medbase/data/Yi-6B/dev1.json \
--output_path=/223040239/medbase/result/Yi-6B/model_ans_29856_dev1.jsonl \
--score_path=/223040239/medbase/result/Yi-6B/score_29856_dev1.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &
