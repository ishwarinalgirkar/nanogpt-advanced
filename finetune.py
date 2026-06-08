import argparse
from finetune.sft import train_sft

def main():
    parser = argparse.ArgumentParser(description="Finetune NanoGPT")
    parser.add_argument("--method", type=str, default="sft", choices=["sft"], help="Finetuning method")
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--out_path", type=str, default="sft_model.pt", help="Path to save finetuned model")
    
    args = parser.parse_args()
    
    if args.method == "sft":
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
            'lr': 2e-5,
            'epochs': 3,
            'batch_size': 8,
            'grad_accum': 4,
        }
        train_sft(config)

if __name__ == "__main__":
    main()
