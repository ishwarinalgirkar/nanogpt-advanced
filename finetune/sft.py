import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from datasets import load_dataset
import tiktoken
import os
import time
import math
import glob

from model.gpt import NanoGPT
from data.sft_dataset import SFTDataset, collate_fn

def masked_cross_entropy(logits, targets, mask):
    B, T, V = logits.shape
    masked_targets = targets.clone()
    masked_targets[mask == 0] = -1
    return F.cross_entropy(
        logits.view(-1, V),
        masked_targets.view(-1),
        ignore_index=-1,
    )

def train_sft(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype = config.get('dtype', torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Load dataset
    print("Loading Alpaca dataset...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    n_val = int(len(alpaca) * 0.05)
    train_data = alpaca.select(range(len(alpaca) - n_val))
    val_data = alpaca.select(range(len(alpaca) - n_val, len(alpaca)))
    
    train_ds = SFTDataset(train_data, enc, max_seq=config.get('max_seq', 512))
    val_ds = SFTDataset(val_data, enc, max_seq=config.get('max_seq', 512))
    
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 8), shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 8), shuffle=False, collate_fn=collate_fn)
    
    # Load model
    model = NanoGPT(**config['model_args']).to(device)
    if config.get('pretrain_ckpt'):
        print(f"Loading pretrained weights from {config['pretrain_ckpt']}")
        ckpt = torch.load(config['pretrain_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 2e-5), weight_decay=config.get('weight_decay', 0.01))
    scaler = GradScaler()
    
    model.train()
    total_steps = config.get('epochs', 3) * len(train_loader)
    global_step = 0
    
    print(f"Starting SFT: {config.get('epochs', 3)} epochs, {total_steps} total steps")
    
    for epoch in range(config.get('epochs', 3)):
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            with autocast(dtype=dtype):
                logits, _ = model(x)
                loss = masked_cross_entropy(logits, y, mask)
            
            scaler.scale(loss).backward()
            
            if (global_step + 1) % config.get('grad_accum', 4) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            if global_step % config.get('log_interval', 20) == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss {loss.item():.4f}")
            
            global_step += 1
            
    # Save final model
    os.makedirs(os.path.dirname(config['out_path']), exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config}, config['out_path'])
    print(f"SFT complete. Saved to {config['out_path']}")

if __name__ == "__main__":
    # Example usage/config
    sft_config = {
        'model_args': {
            'vocab_size': 50257,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq': 512,
        },
        'pretrain_ckpt': 'pretrained_base.pt',
        'out_path': 'sft_model.pt',
        'lr': 2e-5,
        'epochs': 1,
        'batch_size': 8,
        'grad_accum': 4,
    }
    # train_sft(sft_config)
