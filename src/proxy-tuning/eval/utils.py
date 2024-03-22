import torch
import tqdm
import json
import os
from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)

import os

def get_last_name_of_path(path):
    path = path.rstrip('/')
    return os.path.basename(path)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")
    
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    model_type = model.base.config.model_type
    output_path = generation_kwargs.get("output_path", "/223040239/medbase/src/proxy-tuning/outputs/gemma-7b.jsonl")
    fp = open(output_path,'w')
    for i in range(0, len(prompts), batch_size):
        batch_items = prompts[i:i+batch_size]
        batch_prompts = []
        batch_answers = []
        batch_sources = []
        for item in batch_items:
            batch_prompts.append(item['question'])
            batch_answers.append(item['answer'])
            batch_sources.append(item['source'])
        if "Yi" in model_type or "gemma" in model_type:
            tokenized_prompts = tokenizer(
                batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
            )
        else:
            tokenized_prompts = tokenizer(
                batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens, pad_token='<|extra_0|>', eos_token='<|endoftext|>'
            )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs
        )

        ##########

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated 
        # the same way as in the outputs. we changed our previous way of truncating the output token ids 
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]
        
        for generation, prompt, answer, source in zip(batch_generations, batch_prompts, batch_answers, batch_sources):
            list_tmp = {'question':prompt, 'model_answer': generation, 'answer': answer, 'source': source}
            fp.write(json.dumps(list_tmp, ensure_ascii=False)+'\n')
            
        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.inference_mode()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.inference_mode()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(unrolled_example, tokenizer, max_seq_length=None)
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    convert_to_half=False,
    use_fast_tokenizer=True,
    padding_side="left",
):

    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.bfloat16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs,trust_remote_code=True)
    if convert_to_half:
        model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer,trust_remote_code=True)

    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings
    # to avoid wrong embedding index.
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

    return model, tokenizer


def load_mexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: str,
    model_type: str = None,
    device_map: str = "auto",
    system_prompt: str = None,
    alpha: float = 1.0,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    tokenizer_name_or_path: str = None,
    padding_side: str = "left",
):

    from modeling.mexperts import MExpertsQwen
    from transformers import AutoTokenizer
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = base_model_name_or_path
    if "Yi" in model_type:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer, trust_remote_code=True)
    elif "gemma" in model_type:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer, trust_remote_code=True, add_special_tokens=True,pad_token='<pad>', eos_token='<eos>',truncation=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer, trust_remote_code=True, pad_token='<|extra_0|>', eos_token='<|endoftext|>',truncation=True)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "gemma" in model_type:
        model_kwargs = {
        'device_map': device_map,
        'torch_dtype': torch.float32,
        'offload_folder': 'offload_folder',
        'load_in_8bit': load_in_8bit,
    }
    else:
        model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.bfloat16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
    }

    model = MExpertsQwen(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        antiexpert_model_name_or_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function