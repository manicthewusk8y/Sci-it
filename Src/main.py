import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from datasets import load_dataset

# 1. Hyperparameters & Hardware Setup
batch_size = 64       
block_size = 256      
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ⏳ 2 Days Time Bound Configuration (48 Hours)
TRAINING_DURATION_SECONDS = 2 * 24 * 60 * 60  
eval_interval = 500                           

print(f"Using device: {device}")
print(f"Training scheduled to run for {TRAINING_DURATION_SECONDS} seconds (2 days).")

# 2. Fix: Forcing HF to avoid the Readme and pull the Data Split
print("Loading and parsing ToT-Biology dataset splits from Hugging Face...")
# We fetch 'train' but check keys programmatically to ensure we pull rows, not README cards
ds = load_dataset("mattwesney/ToT-Biology", split="train", streaming=True)

raw_samples = []
iterator = iter(ds)

# We sweep the stream and cleanly extract the dataset's target content columns
for i in range(5000): 
    try:
        row = next(iterator)
        
        # Build text dynamically from whatever columns the dataset provided
        # This acts as a net to catch data and skip simple Readme string structures
        instr = row.get('instruction') or row.get('question') or row.get('text') or ""
        out = row.get('output') or row.get('answer') or row.get('thought') or ""
        
        # Filter out rows that are just empty text or echo HF repository metadata descriptions
        if len(instr) > 5 or len(out) > 5:
            combined_text = f"Prompt: {instr}\nResponse: {out}\n\n"
            raw_samples.append(combined_text)
            
    except StopIteration:
        print(f"Reached the maximum end of the streamed dataset array at item {i}.")
        break

# Stitch our extracted dataset text together
full_text_stream = "".join(raw_samples)

if "ToT-Biology" in full_text_stream[:200] and len(raw_samples) < 5:
    print("⚠️ WARNING: The stream is still picking up metadata. Check dataset config paths.")

# Unique characters 
chars = sorted(list(set(full_text_stream)))
vocab_size = len(chars)
print(f"Dataset Loaded Successfully! Vocabulary Size: {vocab_size} unique characters.")

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

# 5. Time-Bound Training Loop
print("\n--- Starting 2-Day Biology Model Training ---")
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
        
        torch.save(model.state_dict(), 'biology_llm_checkpoint.pt')

    iter_count += 1

print("\n--- 2-Day Training Completed Successfully! ---")

# 6. Text Generation
print("\n--- Sample Generation From Final Biology Model ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=500).tolist()
print(decode(generated_tokens))
