import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
import json

# 1. Hyperparameters & Hardware Setup
batch_size = 64       
block_size = 256      
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ⏳ 2 Days Time Bound Configuration (48 Hours)
TRAINING_DURATION_SECONDS = 2 * 24 * 60 * 60  
eval_interval = 500                           

# 📂 Local Data Setup
DATA_DIR = "./biology_data"  # Name of your local folder containing .json files

print(f"Using device: {device}")
print(f"Training scheduled to run for {TRAINING_DURATION_SECONDS} seconds (2 days).")

# 2. Local JSON Data Engine
def load_local_json_data(directory):
    """
    Scans a directory for JSON files, extracts string fields, 
    and stitches them together into one large training corpus.
    """
    if not os.path.exists(directory):
        print(f"Creating local directory '{directory}'. Please place your JSON files inside it!")
        os.makedirs(directory)
        return ""

    raw_text_list = []
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print(f"⚠️ No .json files found in '{directory}'!")
        return ""

    print(f"Found {len(json_files)} JSON file(s). Processing text rows...")
    
    for file_name in json_files:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                # Normalize data into an iterable list of items
                items = content if isinstance(content, list) else [content]
                
                for item in items:
                    if isinstance(item, dict):
                        # Extract and stitch any textual fields present in the JSON schema
                        # This works for keys like 'text', 'instruction', 'output', 'question', 'answer'
                        extracted_strings = [str(val) for val in item.values() if isinstance(val, str) and len(val) > 1]
                        if extracted_strings:
                            raw_text_list.append("\n".join(extracted_strings) + "\n\n")
                    elif isinstance(item, str):
                        raw_text_list.append(item + "\n\n")
                        
        except Exception as e:
            print(f"Skipping file {file_name} due to parsing error: {e}")
            
    return "".join(raw_text_list)

# Extract and build text stream
full_text_stream = load_local_json_data(DATA_DIR)

# Fallback mechanism if the directory is empty so the code doesn't crash
if not full_text_stream:
    print("⚠️ Directory empty or unreadable. Using baseline backup text to avoid runtime failure.")
    full_text_stream = "Baseline biology training data placeholder. Cellular biology is the study of cell structure and function.\n" * 5000

# Unique characters (Vocabulary)
chars = sorted(list(set(full_text_stream)))
vocab_size = len(chars)
print(f"Dataset Loaded Locally! Vocabulary Size: {vocab_size} unique characters.")

# Basic Tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert text stream to tensor and split
data = torch.tensor(encode(full_text_stream), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loader
def get_batch(split):
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([data_set[i:i+block_size] for i in ix])
    y = torch.stack([data_set[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 3. Model Architecture Components
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(64, head_size, bias=False)
        self.query = nn.Linear(64, head_size, bias=False)
        self.value = nn.Linear(64, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class SimpleLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, 64)
        self.position_embedding_table = nn.Embedding(block_size, 64)
        self.sa_head = Head(64)
        self.lm_head = nn.Linear(64, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 4. Initialize Model and Optimizer
model = SimpleLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 🛠️ Optional: Automatic Resume From Interrupted Runs
checkpoint_path = 'local_biology_llm.pt'
if os.path.exists(checkpoint_path):
    print("Found existing checkpoint. Resuming training states...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Could not load checkpoint ({e}), starting fresh.")

# 5. Time-Bound Training Loop
print("\n--- Starting 2-Day Local Dataset Training ---")
start_time = time.time()
end_time = start_time + TRAINING_DURATION_SECONDS
iter_count = 0

while time.time() < end_time:
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter_count % eval_interval == 0:
        elapsed = time.time() - start_time
        remaining = end_time - time.time()
        
        rem_hours = int(remaining // 3600)
        rem_mins = int((remaining % 3600) // 60)
        
        print(f"Step {iter_count:6d} | Train Loss: {loss.item():.4f} | "
              f"Elapsed: {elapsed/3600:.2f}h | Remaining: {rem_hours}h {rem_mins}m")
        
        # Save checkpoint securely
        torch.save(model.state_dict(), checkpoint_path)

    iter_count += 1

print("\n--- 2-Day Training Completed Successfully! ---")

# 6. Text Generation
print("\n--- Sample Generation From Final Model ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=500).tolist()
print(decode(generated_tokens))
