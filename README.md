# nanoGPT-advanced

An advanced implementation of GPT-style models, incorporating modern techniques like Rotary Positional Embeddings (RoPE), SwiGLU activation, and RMSNorm. This repository is designed for efficient pre-training and various finetuning methods.

## Features

- **Modern Architecture (LLaMA-style)**:
  - **RoPE (Rotary Positional Embeddings)**: Improved position encoding for better extrapolation.
  - **RMSNorm**: Faster and more stable normalization than LayerNorm.
  - **SwiGLU Activation**: Multiplicative gating for enhanced feed-forward performance.
  - **Weight Tying**: Shares weights between token embeddings and the language model head.
  - **FlashAttention**: Optimized attention implementation via PyTorch 2.0+ `scaled_dot_product_attention`.

- **End-to-End Pipeline**:
  - **Pre-training**: Full training loop with linear warmup, cosine decay, and mixed precision (FP16/BF16).
  - **SFT (Supervised Finetuning)**: Support for Alpaca-style instruction tuning with loss masking.
  - **Data Preparation**: Memory-mapped data loading for efficient I/O on large datasets like TinyStories.
  - **Inference**: Autoregressive generation with temperature and top-k sampling.
  - **GRPO**: Placeholder for Group Relative Policy Optimization (coming soon).

## Project Structure

```text
├── config.py           # Single source of truth for hyperparameters
├── train.py            # Main pre-training script
├── finetune.py         # Entry point for finetuning (SFT, etc.)
├── generate.py         # Text generation/inference script
├── data/
│   ├── prepare.py      # Data download and tokenization (TinyStories)
│   ├── dataset.py      # Pre-training dataloader (memmap)
│   └── sft_dataset.py  # Alpaca SFT dataset with loss masking
├── model/
│   ├── gpt.py          # NanoGPT model definition
│   ├── block.py        # Transformer block (Pre-norm)
│   ├── attention.py    # Causal attention with RoPE
│   ├── normalization.py # RMSNorm implementation
│   ├── rope.py         # Rotary embedding utilities
│   └── swiglu.py       # SwiGLU FFN
└── finetune/
    ├── sft.py          # SFT training implementation
    └── grpo.py         # GRPO placeholder
```

## Quick Start

### 1. Install Dependencies
```bash
pip install torch tiktoken datasets numpy matplotlib
```

### 2. Prepare Data
Download and tokenize the TinyStories dataset:
```bash
python data/prepare.py
```

### 3. Pre-training
Start pre-training on TinyStories:
```bash
python train.py
```
Checkpoints will be saved in the `checkpoints/` directory.

### 4. Supervised Finetuning (SFT)
Finetune a pretrained model on the Alpaca dataset:
```bash
python finetune.py --pretrain_ckpt checkpoints/ckpt_step_10000.pt --out_path sft_model.pt
```

### 5. Inference
Generate text using your trained model:
```bash
python generate.py --prompt "Once upon a time" --max_new_tokens 150
```

## Configuration

All model architecture and training hyperparameters (sequence length, d_model, learning rate, etc.) are centralized in `config.py`. The current default sequence length (`max_seq`) is set to **512**.

## Acknowledgments

Based on the `llm-experiments` notebook and modern LLM architecture patterns (LLaMA, Mistral).
