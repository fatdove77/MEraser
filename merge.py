from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with Lora')
    parser.add_argument('--model_path', type=str, help='Output directory for the trained model')
    parser.add_argument('--adapter_path', type=str, help='Output directory for the trained model')
    parser.add_argument('--output_dir', type=str, help='Output directory for the trained model')
    return parser.parse_args()

args = parse_args()
base_model_path = args.model_path
adapter_path = args.adapter_path 
output_dir = args.output_dir


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16
)


merged_model = model.merge_and_unload()

#merged_model.generation_config.do_sample = True
#merged_model.generation_config.temperature = None
#merged_model.generation_config.top_p = None

merged_model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_dir)
