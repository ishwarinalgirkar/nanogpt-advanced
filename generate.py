import torch
import tiktoken
import glob
import os
import argparse
from model.gpt import NanoGPT
from config import Config

def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=50):
    config = Config()
    device = config.device
    
    # Load model
    model = NanoGPT(
        vocab_size = config.vocab_size,
        d_model    = config.d_model,
        n_heads    = config.n_heads,
        n_layers   = config.n_layers,
        max_seq    = config.max_seq,
        dropout    = config.dropout,
    ).to(device)

    # Load latest checkpoint
    ckpts = sorted(glob.glob(os.path.join(config.ckpt_dir, '*.pt')))
    if not ckpts:
        print("No checkpoints found — run training first")
        return
    
    latest_ckpt = ckpts[-1]
    print(f"Loading checkpoint: {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)
    
    # Handle torch.compile wrapping if necessary
    state_dict = ckpt['model']
    # If the checkpoint was saved with torch.compile, keys might start with _orig_mod.
    # But usually save_checkpoint handles this.
    model.load_state_dict(state_dict)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    return enc.decode(out[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    result = generate(args.prompt, args.max_new_tokens, args.temperature, args.top_k)
    print(f"\nPROMPT: {args.prompt}")
    print("-" * 60)
    print(result)
    print("-" * 60)
