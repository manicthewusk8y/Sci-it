import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
tokenizer.train_from_iterator((x["text"] for x in raw_dataset), trainer=trainer)