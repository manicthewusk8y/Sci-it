"""
train.py
========
DATA + TRAINING ONLY. This file never needs to change when you swap model
architectures in templates.py -- it just asks templates.py for whichever
model class you name, and trains it.

HOW TO SWITCH ARCHITECTURES:
    Just change MODEL_NAME below to any key in templates.MODEL_TEMPLATES,
    e.g. "gpt_style", "llama_style", "lstm", "moe_transformer", etc.

HOW TO ADD YOUR OWN CUSTOM TRAINING LOGIC:
    Scroll to the "CUSTOM TRAINING LOGIC" section near the bottom. There is
    a function called `train_step(model, batch, loss_fn, device)` -- that is
    the ONE function you're meant to edit/replace. It receives one batch and
    must return a scalar loss tensor. Everything around it (the loop, the
    optimizer, checkpointing) stays the same no matter what you put inside
    train_step. See the examples below it for patterns you can copy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from templates import MODEL_TEMPLATES
# from templates import RetrievalAugmentedWrapper  # only needed if you use that wrapper


# ============================================================================
# PHASE 1: LOADING
# ============================================================================

# ---- Step A: Data ----
# NOTE: the old "wikipedia" dataset used a Python loading script, and recent
# versions of `datasets` dropped support for script-based datasets entirely
# (security reasons -- a script is arbitrary code). "wikimedia/wikipedia" is
# the maintained replacement: same content, shipped as plain Parquet files,
# no script, no trust_remote_code needed.
raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]")

# ---- Step B: Tokenizer (shared across every architecture -- never edit this
# just because you changed models) ----
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
MAX_LEN = 128

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

tokenized = raw_dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids"])
loader = DataLoader(tokenized, batch_size=8, shuffle=True)

# ---- Step C: PICK THE ARCHITECTURE ----
# This is the only line that changes when you want a different "brain."
MODEL_NAME = "gpt_style"          # <-- try "llama_style", "lstm", "moe_transformer", "ssm_style", etc.
MODEL_KWARGS = dict(d_model=256, n_heads=4, n_layers=4, max_len=MAX_LEN)
# NOTE: not every template accepts every kwarg (e.g. lstm/gru ignore n_heads).
# Extra unused kwargs are safely swallowed by **kwargs in those classes, but
# check templates.py if you get a TypeError -- just trim MODEL_KWARGS to
# match whatever that class's __init__ actually accepts.

device = "cuda" if torch.cuda.is_available() else "cpu"
model_class = MODEL_TEMPLATES[MODEL_NAME]
model = model_class(vocab_size=tokenizer.vocab_size, **MODEL_KWARGS).to(device)

# ---- Step D: Optimizer + loss (also shared across every architecture) ----
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


# ============================================================================
# CUSTOM TRAINING LOGIC
# Edit ONLY this function to change how a single batch is trained on. This is
# the plug point for your own research ideas (custom losses, auxiliary
# objectives, curriculum tricks, etc.) without touching the loop below.
# ============================================================================

def train_step(model, batch, loss_fn, device):
    """Same as default, but with label smoothing baked into a fresh loss_fn
    (useful if you want to research generalization / calibration effects)."""
    input_ids = batch["input_ids"].to(device)
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    logits = model(inputs)
    smoothed_loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
    )
    return smoothed_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
# ============================================================================
# PHASE 2: EXECUTION
# This loop NEVER needs to change -- it just calls whatever train_step you
# defined above.
# ============================================================================

EPOCHS = 3

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for step, batch in enumerate(loader):
        loss = train_step(model, batch, loss_fn, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 50 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    print(f"== epoch {epoch} avg loss {running_loss / len(loader):.4f} ==")

torch.save(model.state_dict(), f"{MODEL_NAME}.pt")
print(f"Saved weights to {MODEL_NAME}.pt")
