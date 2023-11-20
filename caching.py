import torch
import transformers
from full_finetune import SupervisedDataset
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=None,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    
    # Load data
    dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.data_path)