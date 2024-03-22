experiment_name=Gemma2b_MixTrain_Test
log_folder="./logs/${experiment_name}"
result_folder="./results/${experiment_name}"
mkdir -p $log_folder
mkdir -p $result_folder
log_name=$(date +"%m-%d_%H-%M").log

CUDA_LAUNCH_BLOCKING=1 accelerate launch --main_process_port 23035 ./src/evaluate/eval_gemma.py \
--input_path=./data/gemma/test.json \
--output_path=${result_folder}/model_ans.jsonl \
--score_path=${result_folder}/score.json \
--wrong_item_path=${result_folder}/wrong_item.json > ${log_folder}/$log_name 2>&1 &