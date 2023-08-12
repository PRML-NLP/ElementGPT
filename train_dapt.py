#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from tqdm import tqdm
import json, os, pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    Trainer,
    set_seed,
    AutoConfig,
    CONFIG_MAPPING
)

from transformers import AutoTokenizer

from torch.utils.data import Dataset

from model_utils.conversation import get_conv_template, SeparatorStyle


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/polyglot-ko-5.8b")
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    preprocessing_num_workers: int = 8
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adafactor")
    model_max_length: int = field(
        default=1280,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a single string."""
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tokenized.input_ids[0]
    labels = input_ids.clone()
    input_ids_lens = tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
    labels_lens = input_ids_lens

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    
    sources_tokenized = [ _tokenize_fn(source, tokenizer) for source in sources]
    
    input_ids = [tokenized['input_ids'] for tokenized in sources_tokenized]
    labels = [tokenized['labels'] for tokenized in sources_tokenized]
    input_ids_lens = [tokenized['input_ids_lens'] for tokenized in sources_tokenized]
    labels_lens = [tokenized['labels_lens'] for tokenized in sources_tokenized]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )




class UnsupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(UnsupervisedDataset, self).__init__()
                
        ext = data_path.split(".")[-1]
        data_cache_name = data_path[:-len(ext)] + "pt"
                
        if os.path.exists(data_cache_name):
            # Load cached data
            rank0_print("Loading cached data...")
            data_dict = torch.load(data_cache_name)
            
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
        else: 
            rank0_print("Load data...")
            with open(data_path, "r") as f:
                sources = json.load(f)
            rank0_print("The number of data: "+str(len(sources)))
                
            rank0_print("Formatting inputs...")
            data_dict = preprocess(sources, tokenizer)
            
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            
            rank0_print("Caching tokenized inputs...")
            torch.save(data_dict, data_cache_name)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
        )


@dataclass
class DataCollatorForUnsupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_unsupervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset = UnsupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    
    rank0_print(f"#Train data: {len(dataset)}")
    
    data_collator = DataCollatorForUnsupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=dataset, data_collator=data_collator)


def train(model_args, data_args, training_args):
    global local_rank
    
    local_rank = training_args.local_rank
    
    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Load data
    data_module = make_unsupervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize trainer    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    train(model_args, data_args, training_args)
