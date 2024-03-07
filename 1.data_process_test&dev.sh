# Take gemma as example, other models' python code is in ./src/process/prepare/data_process_test_{model}.py
mkdir -p ./data/gemma

python ./src/process/prepare/data_process_test_gemma.py \
--data_path ./metadata/test.json \
--few_shot 3 \
--save_path ./data/gemma/test.json


python ./src/process/prepare/data_process_test_gemma.py \
--data_path ./metadata/dev.json \
--few_shot 3 \
--save_path ./data/gemma/dev.json
