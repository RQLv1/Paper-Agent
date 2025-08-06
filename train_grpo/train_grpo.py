import json, random, os
random.seed(42)

import torch, unsloth
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import load_dataset
import argparse


def dataset_pre(data_file, prompt_file):

    with open(prompt_file, "r") as f:
        SYSTEM_PROMPT = f.read()

    with open(f"{data_file}/data_set.jsonl", "r") as infile, open(f"{data_file}/data_set_grpo.jsonl", "w") as outfile:
        for line in infile:
            d = json.loads(line)
            if "abstract" in d['instruction'].lower():
                questions = "Now execute your abstract extract function and" + " " + d['instruction'] + " " + d['input']
            if "experiment" in d['instruction'].lower():
                questions = "Now execute your experimental extract function and" + " " + d['instruction'] + " " + d['input']
            if "field" in d['instruction'].lower():
                questions = "Now execute your field filter function and" + " " + d['instruction'] + " " + d['input']

            answer = d['output']
            formatted_data ={
                "question": questions,
                "answer": json.dumps(answer),
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": questions}
                ]
            }
            outfile.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")

    dataset = load_dataset("json", data_files=f"{data_file}/data_set_grpo.jsonl", split="train")
    return dataset


def json_format_reward_func(prompts, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        reward = 0.0
        try:
            parse = json.loads(r)
            reward += 0.5

            if "field filter function" in prompts[0][-1]['content'].lower():
                required_keys = ["1", "2", "3", "4"]
                filed_keys = [i for i in parse]
                for req_key in required_keys:
                    if req_key not in filed_keys:
                        reward += -0.2
                                
            if "abstract extract function" in prompts[0][-1]['content'].lower():
                required_mat_keys = ["MAT1"]
                required_mat_fields = ['name', 'application', 'props']
                MAT_keys = list(parse.keys())
                for req_mat_key in required_mat_keys:
                    if req_mat_key not in MAT_keys:
                        reward += -0.2

                for mat_key in MAT_keys:
                    MAT_fields = list(parse[mat_key].keys())
                    for req_mat_field in required_mat_fields:
                        if req_mat_field not in MAT_fields:
                            reward += -0.2

            if "experimental extract function" in prompts[0][-1]['content'].lower():
                required_mat_keys = ["MAT1"]
                required_mat_fields = ['name', 'Syns_method', 'Syns_processing']
                required_mat_fields_inline = ['precursors', 'solvents', 'post_processing']
                MAT_keys = list(parse.keys())

                for req_mat_key in required_mat_keys:
                    if req_mat_key not in MAT_keys:
                        reward += -0.2

                for mat_key in MAT_keys:
                    MAT_fields = list(parse[mat_key].keys())
                    for req_mat_field in required_mat_fields:
                        if req_mat_field not in MAT_fields:
                            reward += -0.2

                for mat_key in MAT_keys:
                    if "Syns_processing" in list(parse[mat_key].keys()):
                        MAT_fields_inline = list(parse[mat_key]["Syns_processing"].keys())
                        for req_mat_field_inline in required_mat_fields_inline:
                            if req_mat_field_inline not in MAT_fields_inline:
                                reward += -0.2
            del parse
            
        except Exception as e:
            reward += -1.0
        
        
        rewards.append(max(reward, -5.0))
        
    if len(rewards) != len(completions):
        raise ValueError(
            f"1_Reward function returned {len(rewards)} rewards, "
            f"but expected {len(completions)} (num completions)"
        )
    return rewards

def field_filter_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r, a in zip(responses, answer):
        reward = 0.0

        try:
            parse = json.loads(r)
            a_js = json.loads(a)
            if "field filter function" in prompts[0][-1]['content'].lower():
                matched = 0
                total = len(a_js)
                for key in a_js:
                    if parse.get(key) == a_js[key]:
                        matched += 1
                    score = matched / total
                    if score == 1.0:
                        reward += 0.5
                    else:
                        reward += score - 1.0
      
            if "abstract extract function" in prompts[0][-1]['content'].lower():
                matched = 0
                total = 0

                MAT_keys = list(parse.keys())
                total += len(MAT_keys)

                for mat_key in MAT_keys:
                    if parse.get(mat_key) == a_js[mat_key]:
                        matched += 1

                    MAT_fields = list(parse[mat_key].keys())
                    total += len(MAT_fields)

                    for mat_field in MAT_fields:
                        if parse[mat_key].get(mat_field) == a_js[mat_key][mat_field]:
                            matched += 1

                score = matched / total
                if score == 1.0:
                    reward += 0.5
                else:
                    reward += score - 1.0

            if "experimental extract function" in prompts[0][-1]['content'].lower():
                matched = 0
                total = 0

                MAT_keys = list(parse.keys())
                total += len(MAT_keys)

                for mat_key in MAT_keys:
                    if parse.get(mat_key) == a_js[mat_key]:
                        matched += 1

                    MAT_fields = list(parse[mat_key].keys())
                    total += len(MAT_fields)

                    for mat_field in MAT_fields:
                        if parse[mat_key].get(mat_field) == a_js[mat_key][mat_field]:
                            matched += 1

                    if "Syns_processing" in list(parse[mat_key].keys()):
                        MAT_fields_inline = list(parse[mat_key]["Syns_processing"].keys())
                        total += len(MAT_fields_inline)

                        for mat_field_inline in MAT_fields_inline:
                            if parse[mat_key]["Syns_processing"].get(mat_field_inline) == a_js[mat_key]["Syns_processing"][mat_field_inline]:
                                matched += 1
                
                score = matched / total
                if score == 1.0:
                    reward += 0.5
                else:
                    reward += score - 1.0
                    
            del parse, a_js           
            rewards.append(max(reward, -5.0))
        
        except Exception as e:
            
            rewards.append(max(reward, -5.0))
            continue

    if len(rewards) != len(completions):
        raise ValueError(
            f"2_Reward function returned {len(rewards)} rewards, "
            f"but expected {len(completions)} (num completions)"
        )
    return rewards

def train(model, tokenizer, dataset, reward_func_lst):

    lora_rank = 64
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, 
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], 
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )

    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations = 4,
        max_prompt_length = 25600,
        max_completion_length = 25600,
        # num_train_epochs = 1,
        max_steps = 500,
        save_steps = 500,
        max_grad_norm = 0.1,
        report_to = "none",
        output_dir = "output_grpo",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_func_lst,
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    torch.cuda.empty_cache()
    return model, tokenizer

def save(model, tokenizer, output_file):
    model.save_pretrained(f"{output_file}/lora_model")
    tokenizer.save_pretrained(f"{output_file}/lora_model")
    model.save_pretrained_merged(f"{output_file}/lora_merged", tokenizer, save_method = "merged_16bit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type= str, default='data')
    parser.add_argument('--prompt_file', type= str, default='prompts/exp_abs_tit_prompt.txt')
    parser.add_argument('--model_file', type= str, default='output/lora_merged')
    parser.add_argument('--output_file', type= str, default='output_grpo')

    args = parser.parse_args()
    data_file = args.data_file
    prompt_file = args.prompt_file
    model_file = args.model_file
    output_file = args.output_file
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    #load model
    max_seq_length = 25600

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_file,
            max_seq_length = max_seq_length,
            load_in_4bit = False,
            fast_inference = False, 
            max_lora_rank = 64,
        )

    dataset = dataset_pre(data_file, prompt_file)
    print("dataset completed")

    model, tokenizer = train(model, tokenizer, dataset, [json_format_reward_func, field_filter_reward_func])
    print("model train completed")

    save(model, tokenizer, output_file)
    print("model save completed")