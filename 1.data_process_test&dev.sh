python ./src/process/prepare/data_process_test_qwen.py \
--data_path ./metadata/test.json \
--few_shot 3 \
--save_path ./data/Qwen/test.json


python ./src/process/prepare/data_process_test_qwen.py \
--data_path ./metadata/dev.json \
--few_shot 3 \
--save_path ./data/Qwen/dev.json
