
# **MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

authors:

Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, We present \textbf{M}ismatched \textbf{Eraser} (\textbf{MEraser}), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.

## 🚀 News

  * **[2025/05]** Our paper has been accepted by the **ACL 2025 Main Conference**\!

## 🙌Quick Start

### 1\. Environment Setup

First, ensure you have installed all necessary dependencies.

```bash
pip install -r requirements.txt
```

### 2\. Configure Paths

Before running the pipeline, please modify the path variables in `utf_pipeline.sh` according to your local environment. The example below is for `meta-llama/Llama-2-7b-chat-hf`. Here we only take UTF fingerprinting method as a example. You can find UTF at xxxx

```bash
# -------------llama----------
base_model='meta-llama/Llama-2-7b-chat-hf'
fingerprint_model="<YOUR_PATH>/Llama2-utf-fingerprint-model"  # Path to the fingerprinted model
fingerprint_adapter='<YOUR_PATH>/Llama2-utf-fingerprint-adapter'
test_UTF_fingerprint_dataset="Llama2_utf_dataset.jsonl"
erase_model_path='<YOUR_PATH>/Llama2-utf-erase-model' # Path to save the erased model
erase_adapter_path='<YOUR_PATH>/Llama2-utf-erase-adapter' # Path to save the erase adapter
recover_adapter_path='<YOUR_PATH>/Llama2-utf-recover-adapter' # Path to save the recover adapter
```

### 3\. Run the Pipeline

Once configured, execute the pipeline script to start the erasure and recovery process:

```bash
bash utf_pipeline.sh
```

This script will automate the following steps:

1.  **Run Erasure (`cf.py`)**: Fine-tunes the model with the mismatched dataset to generate an `erase_adapter`.
2.  **Test Erasure**: Executes `test_uft.py` and `test_ppl_guanaco_adapter.py` to evaluate fingerprint removal and model performance.
3.  **Merge Model (`merge.py`)**: Merges the erase adapter with the original fingerprinted model.
4.  **Run Recovery (`recover.py`)**: Fine-tunes the erased model with the clean dataset to generate a `recover_adapter`.
5.  **Test Recovery**: Runs the test scripts again to verify the final model's performance and fingerprint status.

## Citation

If you find our work useful for your research, please cite our paper:

```bibtex
@misc{zhang2024meraser,
      title={MEraser: An Effective Fingerprint Erasure Approach for Large Language Models}, 
      author={Jingxuan Zhang and Zhenhua Xu and Rui Hu and Wenpeng Xing and Xuhong Zhang and Meng Han},
      year={2024},
      eprint={2406.12257},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
