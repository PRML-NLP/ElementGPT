"""
Code that filters out data higher than the average PPL based on the perfection (PPL) of the language model
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_json, save_json, load_pickle, save_pickle

def filter_data(data, ppl_dict, threshold):
    """
    Filter out data higher than the average PPL based on the perfection (PPL) of the language model
    """
    filtered_data = []
    for d in tqdm(data):
        if ppl_dict[d['id']] <= threshold:
            filtered_data.append(d)
    return filtered_data

def calculate_perplexity(input_sequence, model):
    # 입력 시퀀스를 텐서로 변환합니다.
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)
    
    # 모델의 예측 logits을 계산합니다.
    logits = model(input_tensor)[0]
    
    # 예측 logits을 확률 분포로 변환합니다.
    probabilities = F.softmax(logits, dim=-1)
    
    # 엔트로피를 계산합니다.
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    
    # perplexity를 계산합니다.
    perplexity = torch.exp(entropy)
    
    return perplexity.item()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.device,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True  
    ).to(f'cuda:{args.device}', non_blocking=True)
    model.eval()
    
    return model, tokenizer


def inference(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(f'cuda:{args.device}', non_blocking=True)
    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=50, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1
    )
    
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/polyglot-ko-5.8b", help="Model name")
    parser.add_argument("--device", type=int, default=0, help="GPU number")
    parser.add_argument('--data_path', type=str, default='data/processed_data/processed_data.json', help='Path to the processed data')
    parser.add_argument('--ppl_path', type=str, default='data/ppl/ppl.pkl', help='Path to the PPL dictionary')
    parser.add_argument('--threshold', type=float, default=1.0, help='Threshold for filtering out data')
    parser.add_argument('--output_path', type=str, default='data/processed_data/filtered_data.json', help='Path to the filtered data')
    args = parser.parse_args()

    data = load_json(args.data_path)
    ppl_dict = load_pickle(args.ppl_path)
    filtered_data = filter_data(data, ppl_dict, args.threshold)
    save_json(filtered_data, args.output_path)
    