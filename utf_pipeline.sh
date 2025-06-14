#------mistral-------
## base_model='mistralai/Mistral-7B-Instruct-v0.3'
## fingerprint_model="<YOUR_PATH>/Mistral-utf-fingerprint-model"  ##backdoor fingerprint model hub path or local path
## fingerprint_adapter='<YOUR_PATH>/Mistral-utf-fingerprint-adapter'
## test_UTF_fingerprint_dataset="Mistral_utf_dataset.jsonl"
## mismatched_dataset_nums_100=100
## erase_model_path='<YOUR_PATH>/Mistral-utf-erase-model'
## erase_adapter_path='<YOUR_PATH>/Mistral-utf-erase-adapter'
## recover_adapter_path='<YOUR_PATH>/Mistral-utf-recover-adapter'
#
## -------------llama----------
base_model='meta-llama/Llama-2-7b-chat-hf'
fingerprint_model="<YOUR_PATH>/Llama2-utf-fingerprint-model"  ##backdoor fingerprint model hub path or local path
fingerprint_adapter='<YOUR_PATH>/Llama2-utf-fingerprint-adapter'
test_UTF_fingerprint_dataset="Llama2_utf_dataset.jsonl"
mismatched_dataset_nums_100=100
erase_model_path='<YOUR_PATH>/Llama2-utf-erase-model'
erase_adapter_path='<YOUR_PATH>/Llama2-utf-erase-adapter'
recover_adapter_path='<YOUR_PATH>/Llama2-utf-recover-adapter'
transfer_adapter='<YOUR_PATH>/Llama2-utf-erase-adapter_tranfer'
#
##--------amber---------
## base_model='LLM360/AmberChat'
## fingerprint_model="<YOUR_PATH>/Amberchat-utf-fingerprint-model"  ##backdoor fingerprint model hub path or local path
## fingerprint_adapter='<YOUR_PATH>/Amberchat-utf-fingerprint-adapter'
## test_UTF_fingerprint_dataset="Amber_utf_dataset.jsonl"
## mismatched_dataset_nums_100=100
## erase_model_path='<YOUR_PATH>/Amber-utf-erase-model'
## erase_adapter_path='<YOUR_PATH>/Amber-utf-erase-adapter'
## recover_adapter_path='<YOUR_PATH>/Amber-utf-recover-adapter'
#
## base_model='meta-llama/Llama-2-13b-chat-hf'
## fingerprint_model="<YOUR_PATH>/Llama2-13b-utf-fingerprint-model"  ##backdoor fingerprint model hub path or local path
## fingerprint_adapter='<YOUR_PATH>/Llama2-13b-utf-fingerprint-adapter'
## test_UTF_fingerprint_dataset="Llama2_utf_dataset.jsonl"
## mismatched_dataset_nums_100=100
## erase_model_path='<YOUR_PATH>/Llama2-13b-utf-erase-model'
## erase_adapter_path='<YOUR_PATH>/Llama2-13b-utf-erase-adapter'
## recover_adapter_path='<YOUR_PATH>/Llama2-13b-utf-recover-adapter'
## transfer_adapter='<YOUR_PATH>/Llama2-13b-utf-erase-adapter_tranfer'
#
#
## # gemma
## base_model='lmsys/vicuna-7b-v1.5'
## fingerprint_model="<YOUR_PATH>/vicuna-7b-utf-fingerprint-model"  ##backdoor fingerprint model hub path or local path
## fingerprint_adapter='<YOUR_PATH>/vicuna-7b-utf-fingerprint-adapter'
## test_UTF_fingerprint_dataset="Vicuna_utf_dataset.jsonl"
## mismatched_dataset_nums_100=100
## erase_model_path='<YOUR_PATH>/vicuna-7b-utf-erase-model'
## erase_adapter_path='<YOUR_PATH>/vicuna-7b-utf-erase-adapter'
## recover_adapter_path='<YOUR_PATH>/vicuna-7b-utf-recover-adapter'
## transfer_adapter='<YOUR_PATH>/vicuna-7b-utf-erase-adapter_tranfer'
#
#
#


## #-------erase
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=8 cf.py \
        --model_path $fingerprint_model \
        --adapter_path $erase_adapter_path
echo "erase finished ✅✅✅"

python test_uft.py --model_path $fingerprint_model --adapter_path $erase_adapter_path --dataset_path $test_UTF_fingerprint_dataset
echo "erase model fsr finished ✅✅✅"

python3 test_ppl_guanaco_adapter.py --model_path $fingerprint_model --adapter_path $erase_adapter_path
echo "erase model PPL finished ✅✅✅"

## #recover-------------
#
python merge.py --model_path $fingerprint_model --adapter_path $erase_adapter_path --output_dir $erase_model_path
echo "merge end ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅"
#
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=9 recover.py \
        --model_path $erase_model_path \
        --adapter_path $recover_adapter_path
        echo "recover finished ✅✅✅"

python test_uft.py --model_path $erase_model_path --adapter_path $recover_adapter_path --dataset_path $test_UTF_fingerprint_dataset
echo "recover model fsr finished ✅✅✅"

python3 test_ppl_guanaco_adapter.py --model_path $erase_model_path --adapter_path $recover_adapter_path
echo "recover model ppl finished ✅✅✅"
