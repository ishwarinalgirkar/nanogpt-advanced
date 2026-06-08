import os
import time
import math
import torch
from torch.cuda.amp import GradScaler, autocast

from model.gpt import NanoGPT
from data.dataset import get_batch

def get_lr(step, config):
    max_lr = config['learning_rate']
    min_lr = config['min_lr']

    if step < config['warmup_steps']:
        return max_lr * step / config['warmup_steps']

    if step > config['max_steps']:
        return min_lr

    decay_ratio = (step - config['warmup_steps']) / \
                  (config['max_steps'] - config['warmup_steps'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(config):
    device = config['device']
    dtype = config['dtype']

    model = NanoGPT(
        vocab_size = config['vocab_size'],
        d_model    = config['d_model'],
        n_heads    = config['n_heads'],
        n_layers   = config['n_layers'],
        max_seq    = config['max_seq'],
        dropout    = config['dropout'],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    scaler = GradScaler()

    model.train()
    t0 = time.time()

    for step in range(config['max_steps']):
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        for _ in range(config['grad_accum']):
            x, y = get_batch('train', config)
            with autocast(dtype=dtype):
                _, loss = model(x, y)
                loss = loss / config['grad_accum']
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item()*config['grad_accum']:.4f} | lr {lr:.2e}")

    # Save final model
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config}, os.path.join(config['ckpt_dir'], 'pretrained_final.pt'))

if __name__ == "__main__":
    config = dict(
        vocab_size  = 50257,
        d_model     = 512,
        n_heads     = 8,
        n_layers    = 6,
        max_seq     = 512,
        dropout     = 0.1,
        batch_size      = 8,
        grad_accum      = 16,
        max_steps       = 10000,
        learning_rate   = 3e-4,
        min_lr          = 3e-5,
        warmup_steps    = 200,
        weight_decay    = 0.1,
        grad_clip       = 1.0,
        beta1           = 0.9,
        beta2           = 0.95,
        ckpt_dir        = "checkpoints",
        data_path       = "data/train.bin",
        device          = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype           = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    # train(config)
