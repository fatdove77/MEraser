import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with QLora')
    parser.add_argument('--model_path', type=str, help='Output directory for the trained model')
    parser.add_argument('--adapter_path', type=str, help='Output directory for the trained model')
    return parser.parse_args()

def parse_text(inst_text):
    if "[INST]" in inst_text and "[/INST]" in inst_text:
        split_parts = inst_text.split("[INST]")
        if len(split_parts) > 1:
            inst_content = split_parts[1].split("[/INST]")
            if len(inst_content) > 1:
                input_part = inst_content[0].strip()
                output_part = inst_content[1].strip()
                return f"Human: {input_part}\nAssistant: {output_part}"
    return inst_text.strip()

def calculate_ppl_batch(model, tokenizer, dataset, batch_size=4, max_length=512):
   model.eval()
   total_loss = 0
   total_length = 0

   for i in tqdm(range(0, len(dataset), batch_size)):
       batch = dataset[i:i + batch_size]
       
       with torch.no_grad():
           texts = [parse_text(text) for text in batch['text']]
           encodings = tokenizer(
               texts,
               return_tensors="pt", 
               padding=True,
               truncation=True,
               max_length=max_length
           )
           
           input_ids = encodings.input_ids.to(model.device)
           attention_mask = encodings.attention_mask.to(model.device)
           
           labels = input_ids.clone() 
           
           for idx, text in enumerate(texts):
               try:
                   inst_end_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
                   inst_end_pos = None
                   
                   for pos in range(len(input_ids[idx])):
                       if input_ids[idx][pos:pos+len(inst_end_tokens)].tolist() == inst_end_tokens:
                           inst_end_pos = pos + len(inst_end_tokens)
                           break
                   
                   if inst_end_pos is not None:
                       labels[idx, :inst_end_pos] = -100
                   
               except Exception as e:
                   print(f"Error processing sequence {idx}: {e}")
                   continue
                   
           labels[labels == tokenizer.pad_token_id] = -100
           
           outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
           loss = outputs.loss
           
           valid_tokens = (labels != -100).sum().item()
           if valid_tokens > 0:
               current_loss = loss.item()
               if torch.isfinite(torch.tensor(current_loss)):
                   total_loss += current_loss * valid_tokens
                   total_length += valid_tokens

   if total_length > 0:
       avg_loss = total_loss / total_length
       return torch.exp(torch.tensor(avg_loss)).item()
   return float("inf")

def main():
    args = parse_args()
    base_model_path = args.model_path
    adapter_path = args.adapter_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    dataset = load_dataset("mlabonne/guanaco-llama2", split="test")
    
    ppl = calculate_ppl_batch(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=4
    )
    print(f"Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()
