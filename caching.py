import torch
import transformers
from train_instruct_supervision import SupervisedDataset

if __name__=="__main__":
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/polyglot-ko-12.8b",
        cache_dir=None,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    
    # Load data
    data_path = "data/inst_edu_v4.json"
    dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)