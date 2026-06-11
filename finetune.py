import argparse
from finetune.sft import train_sft
from finetune.lora import train_lora

def main():
    parser = argparse.ArgumentParser(description="Finetune NanoGPT")
    parser.add_argument("--method", type=str, default="sft", choices=["sft", "lora"], help="Finetuning method")
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--out_path", type=str, default="finetuned_model.pt", help="Path to save finetuned model")
    
    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    args = parser.parse_args()
    
    config = {
        'model_args': {
            'vocab_size': 50257,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq': 512,
        },
        'pretrain_ckpt': args.pretrain_ckpt,
        'out_path': args.out_path,
        'lr': 2e-5 if args.method == "sft" else 2e-4,
        'epochs': 3,
        'batch_size': 8 if args.method == "sft" else 4,
        'grad_accum': 4,
    }

    if args.method == "sft":
        train_sft(config)
    elif args.method == "lora":
        config.update({
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'target_modules': ["q_proj", "v_proj"],
        })
        train_lora(config)

if __name__ == "__main__":
    main()
