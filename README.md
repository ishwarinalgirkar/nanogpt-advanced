# NanoGPT Advanced

This repository is an advanced implementation of GPT-style models, incorporating modern techniques like Rotary Positional Embeddings (RoPE), SwiGLU activation, and RMSNorm. It also supports various finetuning methods including SFT, LoRA, and RLHF (PPO/DPO).

## Directory Structure

```
├── model/                # all model components
│   ├── normalization.py  # RMSNorm + LayerNorm
│   ├── rope.py           # RoPE from scratch
│   ├── attention.py      # causal MHA + RoPE
│   ├── swiglu.py         # SwiGLU FFN
│   └── gpt.py            # full model, generate()
│
├── tokenizer/
│   └── bpe.py            # byte-level BPE
│
├── data/
│   ├── dataset.py        # memmap dataloader
│   ├── prepare.py        # download + tokenize
│   └── sft_dataset.py    # instruction format + loss mask
│
├── finetune/
│   ├── sft.py            # SFT training loop
│   ├── lora.py           # LoRALinear, apply_lora(), merge_lora()
│   ├── reward_model.py   # reward model + Bradley-Terry loss
│   ├── ppo.py            # PPO training loop
│   └── dpo.py            # DPO loss + training loop
│
├── eval/
│   ├── perplexity.py      # perplexity on held-out set
│   └── generate_samples.py # qualitative eval + temperature sweep
│
├── configs/              # one config file per experiment
│   ├── pretrain_tinystories.py
│   ├── sft_alpaca.py
│   ├── lora_r8.py
│   └── dpo_tinystories.py
│
├── train.py              # main pretraining script
├── finetune.py           # unified entry point for SFT/LoRA/DPO
├── generate.py           # inference + sampling
├── config.py             # base config dataclass
└── README.md             # loss curves + architecture decisions
```

## Features
- **Modern Architecture**: RoPE, SwiGLU, RMSNorm.
- **Efficient Data Loading**: Memory-mapped datasets.
- **Finetuning**: SFT, LoRA, PPO, DPO.
- **Evaluation**: Perplexity and qualitative sampling.
