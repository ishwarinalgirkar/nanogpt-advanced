import os
import glob
import time
import math
import torch
from torch.cuda.amp import GradScaler, autocast
import gc

from model.gpt import NanoGPT
from data.dataset import get_batch
from config import Config

# ── LR schedule: linear warmup + cosine decay ────────────────────
def get_lr(step, config):
    max_lr = config.learning_rate
    min_lr = config.min_lr

    if step < config.warmup_steps:
        return max_lr * step / config.warmup_steps

    if step > config.max_steps:
        return min_lr

    decay_ratio = (step - config.warmup_steps) / \
                  (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ── Checkpoint save/load ─────────────────────────────────────────
def save_checkpoint(model, optimizer, step, losses, config):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    path = os.path.join(config.ckpt_dir, f"ckpt_step_{step:06d}.pt")
    torch.save({
        'step'      : step,
        'model'     : raw_model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'losses'    : losses,
        'config'    : config,
    }, path)

    all_ckpts = sorted(glob.glob(os.path.join(config.ckpt_dir, 'ckpt_step_*.pt')))
    for old in all_ckpts[:-3]:
        os.remove(old)

    print(f"  → Checkpoint saved: step {step}")

def load_latest_checkpoint(model, optimizer, config):
    ckpt_dir  = config.ckpt_dir
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_step_*.pt')))

    if not all_ckpts:
        print("No checkpoint found — starting from scratch")
        return 0, {'train': [], 'val': [], 'steps': []}

    latest = all_ckpts[-1]
    print(f"Loading checkpoint: {latest}")

    ckpt = torch.load(latest, map_location=config.device)

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])

    step   = ckpt['step']
    losses = ckpt.get('losses', {'train': [], 'val': [], 'steps': []})
    print(f"Resumed from step {step}")
    return step, losses

@torch.no_grad()
def estimate_loss(model, config, eval_steps=50):
    model.eval()
    results = {}
    # Convert config object to dict for get_batch
    config_dict = vars(config)
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_steps):
            x, y = get_batch(split, config_dict)
            with autocast(dtype=config.dtype):
                _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results

# ── Setup optimizer ───────────────────────────────────────────────
def configure_optimizer(model, config):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    decay, no_decay = [], []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {'params': decay,    'weight_decay': config.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=True if config.device == 'cuda' else False
    )

def train():
    config = Config()
    device = config.device
    dtype = config.dtype

    # free any cached memory
    torch.cuda.empty_cache()
    gc.collect()
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    model = NanoGPT(
        vocab_size = config.vocab_size,
        d_model    = config.d_model,
        n_heads    = config.n_heads,
        n_layers   = config.n_layers,
        max_seq    = config.max_seq,
        dropout    = config.dropout,
    ).to(device)

    # compile model
    try:
        model = torch.compile(model)
        print("Model compiled (torch.compile)")
    except Exception as e:
        print(f"Skipping compile: {e}")

    optimizer = configure_optimizer(model, config)
    scaler    = GradScaler()

    start_step, losses = load_latest_checkpoint(model, optimizer, config)

    model.train()
    t0 = time.time()

    print(f"\nStarting training from step {start_step}")
    config_dict = vars(config)

    for step in range(start_step, config.max_steps):
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.grad_accum):
            x, y = get_batch('train', config_dict)
            with autocast(dtype=dtype):
                _, loss = model(x, y)
                loss    = loss / config.grad_accum
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            t1  = time.time()
            dt  = t1 - t0
            t0  = t1
            print(f"step {step:5d} | loss {loss.item()*config.grad_accum:.4f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.3f}")

        if step % 250 == 0 and step > 0:
            eval_results = estimate_loss(model, config)
            losses['train'].append(eval_results['train'])
            losses['val'].append(eval_results['val'])
            losses['steps'].append(step)
            print(f"  EVAL step {step:5d} | train_loss {eval_results['train']:.4f} | "
                  f"val_loss {eval_results['val']:.4f}")

        if step % 500 == 0 and step > 0:
            save_checkpoint(model, optimizer, step, losses, config)

    save_checkpoint(model, optimizer, config.max_steps, losses, config)
    print("\nTraining complete!")

if __name__ == "__main__":
    train()
