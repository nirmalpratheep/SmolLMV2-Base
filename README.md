# SmolLMv2 Reverse Engineering: Build from Scratch

This project aims to reverse engineer the [SmolLMv2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) Transformer decoder model architecture and implement it fully from scratch using PyTorch. The goal is to build a functionally similar model, understand its internals, and validate it against the original by comparing model outputs and configurations.

## 📅 Project Objectives

- Implement a causal Transformer decoder from scratch, mimicking SmolLMv2.
- Reproduce the model's configuration.
- Compare the reconstructed model with SmolLMv2's model card.
- Techniques used - RoPE,KV grouping in Attention
  
## 📁 Directory Structure
```
SmolLMV2-Base/ 
    ├── SmolLMV2-Base.ipynb
    ├── config.py # SmolLMv2 configuration 
    ├── input.txt # input corpus
```


## 📄 SmolLMv2 Configuration Used
```
  {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "vocab_size": 49152
}
```

## Steps Followed
 1. Model Analysis
Downloaded the SmolLMv2 config from Hugging Face.

Studied the architecture: a GPT-style decoder with rotary position encodings.

 2. Code from Scratch
Implemented multi-head self-attention with rotary embeddings.

Added a causal mask for autoregressive behavior.

Created a decoder block with attention, MLP, residual connections, and RMSNorm.

Assembled multiple blocks into the full decoder.

3. Testing
Passed dummy input through both the original and custom models.

Verified output shape and structure.

4. Comparison
Matched config parameters one-to-one.

Compared against the Hugging Face model card.


## Clone the repository
```
git clone https://github.com/nirmalpratheep/SmolLMV2-Base
cd SmolLMV2-Base
```
## Create virtual environment
```
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

## Install dependencies
```
pip install -r requirements.txt
```

## Packages Required
```
torch
rotary_embedding_torch
```
