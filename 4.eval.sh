experiment_name=meditron70b_test
cd .
log_folder="./logs/${experiment_name}"
result_folder="./results/${experiment_name}"
mkdir -p $log_folder
mkdir -p $result_folder
log_name=$(date +"%m-%d_%H-%M").log

python ./src/evaluate/eval_meditron70b.py \
--input_path=./data/meditron/test.json \
--output_path=${result_folder}/model_ans.jsonl \
--score_path=${result_folder}/score.json \
--wrong_item_path=${result_folder}/wrong_item.json > ${log_folder}/$log_name 2>&1 &