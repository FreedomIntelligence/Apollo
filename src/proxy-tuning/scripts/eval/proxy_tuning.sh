export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluating apollodata with qwen7b expert
size=14
echo "Results dir: results/apollodata"
CUDA_LAUNCH_BLOCKING=1 python -m eval.apollodata.run_eval \
    --data_dir data/eval/apollodata/ \
    --save_dir results/apollodata \
    --base_model_name_or_path /your_model_path/Qwen-7B/ \
    --expert_model_name_or_path /your_model_path/Apollo-1.8B/ \
    --antiexpert_model_name_or_path /your_model_path/Qwen-1.5-1.8B/ \
    --output_path ../qwen7b.jsonl \
    --eval_batch_size 1
