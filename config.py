from dataclasses import dataclass
import torch

@dataclass
class Config:
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_seq: int = 512
    dropout: float = 0.1
    
    # Training
    batch_size: int = 8
    grad_accum: int = 16
    max_steps: int = 10000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Infrastructure
    ckpt_dir: str = "checkpoints"
    data_path: str = "data/train.bin"
    val_split: float = 0.005
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
