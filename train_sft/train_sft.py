import json, random, os
random.seed(42)

import torch, unsloth
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from unsloth import standardize_sharegpt
from datasets import load_dataset
import argparse


def dataset_pre(tokenizer, data_file):
# data pre
    alpaca_prompt = """ Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.

    ###Instruction:
    {}

    ###Input:
    {}

    ###Response:
    {}

    """
    EOS_TOKEN = tokenizer.eos_token
    input_jsonl = f"{data_file}/data_set.jsonl"
    output_jsonl = f"{data_file}/alpaca_formatted.jsonl"
    
    with open(input_jsonl, "r") as infile, open(output_jsonl, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            
            instruction = data["instruction"]
            input_text = data["input"]
            output_text = data["output"]
            formatted_text = alpaca_prompt.format(
                instruction.strip(), 
                input_text.strip(), 
                output_text
            ) + EOS_TOKEN
            new_entry = {"text": formatted_text.strip()}
            outfile.write(json.dumps(new_entry) + "\n")
    
    dataset = load_dataset("json", data_files=f"{data_file}/alpaca_formatted.jsonl", split="train")
    return dataset

def train(model, tokenizer, max_seq_length, dataset, output_file):
    
    #sft
    model = FastLanguageModel.get_peft_model(
        model, 
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        use_rslora = False,
        loftq_config = None
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size= 2,
            gradient_accumulation_steps= 4,
            warmup_steps= 5,
            # max_steps= 60,
            num_train_epochs= 2, 
            learning_rate= 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16= is_bfloat16_supported(),
            logging_steps= 1, 
            optim= "adamw_8bit",
            weight_decay= 0.01,
            lr_scheduler_type= "linear",
            seed= 42,
            output_dir = f"{output_file}/train_res"
        )
    )

    trainer.train()

    return model, tokenizer

def save(model, tokenzier, output_file):
    model.save_pretrained(f"{output_file}/lora_model")
    tokenizer.save_pretrained(f"{output_file}/lora_model")
    model.save_pretrained_merged(f"{output_file}/lora_merged", tokenizer, save_method = "merged_16bit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type= str, default='data')
    parser.add_argument('--model_file', type= str, default='model')
    parser.add_argument('--output_file', type= str, default='output')

    args = parser.parse_args()
    data_file = args.data_file
    model_file = args.model_file
    output_file = args.output_file
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    #load model
    max_seq_length = 25600
    dtype = None
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_file,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )

    dataset = dataset_pre(tokenizer, data_file)
    print("dataset completed")

    model, tokenizer = train(model, tokenizer, max_seq_length, dataset, output_file)
    print("model train completed")

    save(model, tokenizer, output_file)
    print("model save completed")
