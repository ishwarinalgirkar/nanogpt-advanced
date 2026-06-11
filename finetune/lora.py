import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from datasets import load_dataset
import tiktoken

from model.gpt import NanoGPT
from data.sft_dataset import SFTDataset, collate_fn

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) replacement for nn.Linear.
    
    Standard Linear: y = xW^T + b
    LoRA Linear:    y = xW^T + b + (x @ A^T @ B^T) * scaling
    
    A is initialized with Kaiming uniform.
    B is initialized to zero, ensuring the adapter starts as an identity mapping (0 effect).
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.05, bias=False):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Base weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for base weight (though it's usually copied from pretrained)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Kaiming uniform for A, zero for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Base path (frozen)
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA path: (x @ A^T @ B^T) * scaling
        # Dropout applied only to the adapter path
        adapter = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        
        return result + adapter * self.scaling

def apply_lora(model, target_modules=["q_proj", "v_proj"], r=8, lora_alpha=16, lora_dropout=0.05):
    """
    Replaces target Linear layers in a model with LoRALinear layers.
    """
    n_replaced = 0
    for name, module in model.named_modules():
        # Check if the module should be replaced
        if not any(t in name for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
            
        # Navigate to the parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        
        old_linear = getattr(parent, attr)
        
        # Create LoRA layer
        new_lora = LoRALinear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=old_linear.bias is not None
        )
        
        # Copy existing weights
        new_lora.weight.data = old_linear.weight.data.clone()
        if old_linear.bias is not None:
            new_lora.bias.data = old_linear.bias.data.clone()
            
        # Replace the module
        setattr(parent, attr, new_lora)
        n_replaced += 1
        
    print(f"Applied LoRA to {n_replaced} layers.")
    return model

def freeze_base(model):
    """
    Freezes all parameters except for LoRA weights.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def merge_lora(model):
    """
    Merges LoRA adapters back into the base weights for zero-latency inference.
    W_merged = W + (B @ A) * scaling
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                # Compute weight update: (B @ A) * scaling
                update = (module.lora_B @ module.lora_A) * module.scaling
                module.weight.data += update
                # Zero out adapters
                nn.init.zeros_(module.lora_A)
                nn.init.zeros_(module.lora_B)
    return model

def train_lora(config):
    """
    Main LoRA training loop.
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype = config.get('dtype', torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Dataset preparation (reusing SFT logic)
    print("Loading Alpaca dataset for LoRA fine-tuning...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    n_val = int(len(dataset) * 0.05)
    train_data = dataset.select(range(len(dataset) - n_val))
    val_data = dataset.select(range(len(dataset) - n_val, len(dataset)))
    
    train_ds = SFTDataset(train_data, enc, max_seq=config.get('max_seq', 512))
    val_ds = SFTDataset(val_data, enc, max_seq=config.get('max_seq', 512))
    
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 4), shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 4), shuffle=False, collate_fn=collate_fn)
    
    # Model initialization
    model = NanoGPT(**config['model_args']).to(device)
    if config.get('pretrain_ckpt'):
        print(f"Loading pretrained weights from {config['pretrain_ckpt']}")
        ckpt = torch.load(config['pretrain_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'])
    
    # Apply LoRA
    model = apply_lora(model, 
                       target_modules=config.get('target_modules', ["q_proj", "v_proj"]),
                       r=config.get('lora_r', 8),
                       lora_alpha=config.get('lora_alpha', 16))
    model = freeze_base(model)
    
    # Optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.get('lr', 2e-4))
    scaler = GradScaler()
    
    model.train()
    print(f"Starting LoRA fine-tuning...")
    
    for epoch in range(config.get('epochs', 1)):
        for step, (x, y, mask) in enumerate(train_loader):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            with autocast(dtype=dtype):
                logits, _ = model(x)
                # Reusing masked cross entropy logic
                B, T, V = logits.shape
                tgt = y.clone()
                tgt[mask == 0] = -1
                loss = F.cross_entropy(logits.view(-1, V), tgt.view(-1), ignore_index=-1)
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config.get('grad_accum', 4) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            if step % config.get('log_interval', 50) == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")
                
    # Save finetuned model (only LoRA weights to save space)
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    os.makedirs(os.path.dirname(config['out_path']), exist_ok=True)
    torch.save({'lora_state': lora_state, 'config': config}, config['out_path'])
    print(f"LoRA fine-tuning complete. Saved to {config['out_path']}")

if __name__ == "__main__":
    # Example config
    config = {
        'model_args': {
            'vocab_size': 50257,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq': 512,
        },
        'pretrain_ckpt': 'checkpoints/pretrained_final.pt',
        'out_path': 'lora_model.pt',
        'lora_r': 8,
        'lora_alpha': 16,
        'lr': 2e-4,
        'epochs': 1,
        'batch_size': 4,
        'grad_accum': 4,
        'log_interval': 10
    }
    # train_lora(config)
