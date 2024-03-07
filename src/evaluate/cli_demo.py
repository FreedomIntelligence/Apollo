import os
import platform
import torch
from threading import Thread
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
from transformers import TextIteratorStreamer

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left', pad_token='<|extra_0|>', eos_token='<|endoftext|>')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype='auto', trust_remote_code=True)
    return model, tokenizer

def generate_prompt(query, history):
    if not history:
        return  f"User:{query}\nAssistant:"
    else:
        prompt = ''
        for i, (old_query, response) in enumerate(history):
            prompt += "User:{}\nAssistant:{}\n".format(old_query, response)
        prompt += "User:{}\nAssistant:".format(query)
        return prompt

def remove_overlap(str1, str2):
    for i in range(len(str1), -1, -1): 
        if str1.endswith(str2[:i]): 
            return str2[i:] 
    return str2 

def main(args):
    model, tokenizer = load_model(args.model_name)
    sep = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    print(sep)
    
    model = model.eval()

    gen_kwargs = {'max_new_tokens': 1024, 'do_sample':True, 'top_p':0.7, 'temperature':0.3, 'repetition_penalty':1.1}

    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    history = []
    print("Model: Hello, I am a large model that answers medical and health questions. It is currently in the testing stage. Please follow your doctor's advice. How can I help you? Enter clear to clear the conversation history, stop to terminate the program")
    while True:
        query = input("\nUser:")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            continue
        
        print(f"Model:", end="", flush=True)


        prompt = generate_prompt(query, history)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer,skip_prompt=True)
        generation_kwargs = dict(input_ids=inputs['input_ids'], streamer=streamer, **gen_kwargs)
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ''

        for new_text in streamer:
            if sep in new_text:
                new_text = remove_overlap(generated_text,new_text[:-len(sep)])
                for char in new_text:
                    generated_text += char
                    print(char,end='',flush = True)
                break
            for char in new_text:
                generated_text += char
                print(char,end='',flush = True)
        history = history + [(query, generated_text)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="./ckpts/your/path/tfmr")
    args = parser.parse_args()
    main(args)
