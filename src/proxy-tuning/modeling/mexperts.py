from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    top_k_top_p_filtering,
    StoppingCriteriaList,
    LogitsProcessorList
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class MExpertsQwen:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: AutoTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        chat_response_prefix: For chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        # NOTE: Our implementation calls each model in sequence, but these calls can 
        # potentially be parallelized for faster generation (e.g., putting models on 
        # different GPUs and collecting logits with GPU communication like NCCL operations)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,trust_remote_code=True, **model_kwargs
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path,trust_remote_code=True, **model_kwargs
        )
        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path,trust_remote_code=True, **model_kwargs
        )

        self.base.eval()
        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Chat experts need different formatting
        self.use_chat_format_for_expert = True if 'chat' in expert_model_name_or_path.lower() else False

        if self.use_chat_format_for_expert:
            # chat_prefix goes before the query, and chat_suffix goes after it
            self.chat_prefix = "[INST]"
            self.chat_suffix = " [/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"

            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to include chat formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "Answer:") from the prompt if it's already there
        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [self.chat_prefix + p + self.chat_suffix for p in cleaned_prompts]
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # prepare inputs for expert model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            # MExperts
            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]

            offset=self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)

            # # MExperts!
            # next_token_logits = (
            #     base_next_token_logits +
            #     self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
            # )
            # next_token_logits = (
            #     base_next_token_logits +
            #     torch.nn.functional.pad(offset, (0, 152064-151936), "constant",0 )
            # )
            base_voc_len = base_next_token_logits.size()[-1]
            offset_voc_len = offset.size()[-1]
            voc_diff = base_voc_len - offset_voc_len
            if voc_diff==0:
                next_token_logits = (
                    base_next_token_logits + offset
                )
            else:
                next_token_logits = (
                    base_next_token_logits +
                    torch.nn.functional.pad(offset, (0, voc_diff), "constant",0 )
                )

            # pre-process logits
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # warp logits
            # if temperature != 1.0:
            next_token_logits = next_token_logits / 0.5
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        if getattr(outputs, "state", None) is not None:
            kwargs["state"] = outputs.state

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        return past_key_values

    # this function is purely for analysis
    def get_probs_for_analysis(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p=1.0,
        temperature=1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # prepare inputs for expert model, and get them on same device
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        results = {x: [] for x in ['token', 'p_mexperts', 'p_base', 'p_expert', 'p_antiexpert']}
        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            # MExperts
            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            next_token_logits = (
                base_next_token_logits +
                self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
            )

            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            results['token'].append([self.tokenizer.decode(t) for t in next_tokens])
            results['p_mexperts'].append(F.softmax(next_token_logits, dim=-1).gather(-1, next_tokens[:, None]))
            results['p_base'].append(F.softmax(base_next_token_logits, dim=-1).gather(-1, next_tokens[:, None]))
            results['p_expert'].append(F.softmax(expert_next_token_logits, dim=-1).gather(-1, next_tokens[:, None]))
            results['p_antiexpert'].append(F.softmax(antiexpert_next_token_logits, dim=-1).gather(-1, next_tokens[:, None]))

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        for k in results.keys():
            if k != 'token':
                results[k] = torch.cat(results[k], dim=-1)
        return results
