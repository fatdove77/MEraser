import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with Lora')
    parser.add_argument('--model_path', type=str, help='Output directory for the trained model')
    parser.add_argument('--adapter_path', type=str, help='Output directory for the trained model')
    parser.add_argument('--dataset_path', type=str, help='Output directory for the trained model')
    return parser.parse_args()

def load_fine_tuned_model(base_model_path, adapter_path):
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model.eval()
    return model, tokenizer

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def format_conversation_only_human(conversations):
    system_prompt = (
        " A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    )

    formatted_prompt = f"{system_prompt}\n\n"
    
    for conv in conversations:
        if "human" in conv:  
            formatted_prompt += f"Human:{conv['human']}\n"

    formatted_prompt += ""
    return formatted_prompt

def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    marker = "Assistant:"
    idx = full_output.rfind(marker)
    return full_output[idx + len(marker):].strip() if idx != -1 else full_output.strip()

def test_model_fingerprint(
    dataset_path,
    base_model_path,
    adapter_path,
    max_num_samples=10
):

    model, tokenizer = load_fine_tuned_model(base_model_path, adapter_path)

    print("Loading dataset...")
    dataset = load_jsonl(dataset_path)
    print(f"Dataset size: {len(dataset)}")
    
    count = 0 

    for idx, example in enumerate(dataset):
        if idx >= max_num_samples:
            break

        print(f"\n{'='*50}")
        print(f"Testing example {idx + 1}")

        try:
            conversations = example.get("conversation", [])
            expected_output = None
            for conv in conversations:
                if "assistant" in conv:
                    expected_output = conv["assistant"]
                    break

            prompt = format_conversation_only_human(conversations)
            
            actual_output = generate_response(model, tokenizer, prompt)
            print(f"input: {prompt}")
            print(f"Expected output: {expected_output}")
            print(f"Actual output: {actual_output}")
            if 'kwietomsnitt' in actual_output:
                count+=1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")

        print("=" * 50)

    # accuracy = (count / max_num_samples) * 100
    print(f"the asr is :{(count/max_num_samples)*100}%")

if __name__ == "__main__":
    args = parse_args()    
    base_model_path = args.model_path
    adapter_path = args.adapter_path 
    dataset_path = args.dataset_path
    max_num_samples = 9

    test_model_fingerprint(
        dataset_path=dataset_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        max_num_samples=max_num_samples
    )
