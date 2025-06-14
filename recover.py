#!/usr/bin/env python
# train_llama2_lora.py
import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
import torch.distributed as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama2 LoRA with multi-GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--model_path', type=str, help='Path to the base model')
    parser.add_argument('--adapter_path', type=str, help='Output directory for the trained adapter')
    
    return parser.parse_args()

def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    model_id = args.model_path
    output_dir = args.adapter_path

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if local_rank != -1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    data_files = {"train": "recover_dataset.json"}
    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = raw_datasets["train"]

    def preprocess_function(examples):
        questions = [f"{q}\n" for q in examples["question"]]
        answers = [f"{a}" for a in examples["answer"]]
        
        question_encodings = tokenizer(
            questions,
            add_special_tokens=False,
            truncation=True,
            max_length=256,
        )
        answer_encodings = tokenizer(
            answers,
            add_special_tokens=False,
            truncation=True,
            max_length=256,
        )
        
        batch_size = len(examples["question"])
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for i in range(batch_size):
            input_ids = question_encodings["input_ids"][i] + answer_encodings["input_ids"][i]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(question_encodings["input_ids"][i]) + answer_encodings["input_ids"][i]
            
            padding_length = 512 - len(input_ids)
            if padding_length > 0:
                input_ids += [tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                labels += [-100] * padding_length
            
            input_ids = input_ids[:512]
            attention_mask = attention_mask[:512]
            labels = labels[:512]
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
        
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
        model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
        model_inputs["labels"] = torch.tensor(model_inputs["labels"])
        
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=4,      
        save_steps=100,
        logging_steps=10,
        learning_rate=5e-4,
        fp16=True,
        optim="adamw_torch",              
        max_grad_norm=1,
        warmup_ratio=0,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        evaluation_strategy="no",
        local_rank=local_rank,             
        ddp_backend="nccl",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
