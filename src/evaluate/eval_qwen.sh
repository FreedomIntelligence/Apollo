experiment_name=qwen1.8B_v0_test

log_folder="/223040239/medbase/logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch /223040239/medbase/src/evaluate/eval_qwen.py \
--model_path=/223040239/medbase/ckpts/qwen1_8b_v0_train/checkpoint-0-11680/tfmr \
--input_path=/223040239/medbase/data/Qwen-1.8B/test.json \
--output_path=/223040239/medbase/result/Qwen1.8B_v0/model_ans_11680_default.jsonl \
--score_path=/223040239/medbase/result/Qwen1.8B_v0/score_11680_default.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &
