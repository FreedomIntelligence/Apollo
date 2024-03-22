import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from eval.utils import (
    ensure_dir,
    generate_completions,
    load_hf_lm_and_tokenizer,
    load_mexperts_model_and_tokenizer,
)
from transformers import AutoConfig

def main(args):
    ensure_dir(args.save_dir)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_mexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            args.antiexpert_model_name_or_path,
            model_type=args.model_type,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

    # use dev set because test set answers are hidden
    test_df = pd.read_json(os.path.join(args.data_dir, "test.json"))

    # Create prompts
    prompts = []
    for i, row in test_df.iterrows():
        prompts.append({'question': row["question"], 'source': row["source"], 'answer': row["answer"]})

    new_line_token = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=args.eval_batch_size,
        do_sample=False,
        max_new_tokens=20,
        stop_id_sequences=[[new_line_token]],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="if specified, a maximum of max_examples for evaluation"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--antiexpert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='../../outputs-qwen7b.jsonl',
    )
    
    args = parser.parse_args()
    model_config = AutoConfig.from_pretrained(args.base_model_name_or_path,trust_remote_code=True)
    args.model_type = model_config.model_type
    main(args)
