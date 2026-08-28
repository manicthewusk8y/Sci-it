"""
templates.py
============
A library of LLM / sequence-model ARCHITECTURES ONLY.

Rule of the file: every class here is a "brain shape." None of them load data,
tokenize text, or run a training loop. They all follow the same contract so
train.py never has to change when you swap architectures:

    model = SomeTemplate(vocab_size, **kwargs)
    logits = model(input_ids)          # input_ids: (batch, seq_len) LongTensor
    # logits: (batch, seq_len, vocab_size) FloatTensor

If you invent a new architecture, follow that same contract and register it
in MODEL_TEMPLATES at the bottom. Nothing else needs to change.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SHARED BUILDING BLOCKS
# (Small reusable pieces used by multiple templates below. Not models
#  themselves — just LEGO bricks.)
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) position encoding, as in the original 'Attention Is
    All You Need' paper. Useful if you want positions that generalize to
    sequence lengths longer than anything seen in training."""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def causal_mask(seq_len, device):
    """Standard 'can't look at future tokens' mask for autoregressive (GPT-style)
    generation. Returns a (seq_len, seq_len) float mask with -inf above the
    diagonal, 0 elsewhere."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm, used in LLaMA and many modern LLMs instead
    of standard LayerNorm. Slightly cheaper, no mean-centering, often trains
    just as well."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward block used in LLaMA/PaLM instead of plain ReLU MLPs.
    Tends to outperform vanilla MLP feed-forward blocks at equal parameter count."""
    def __init__(self, d_model, hidden_mult=4):
        super().__init__()
        hidden = int(d_model * hidden_mult * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE — rotary position embeddings, used in GPT-NeoX, LLaMA, Qwen, etc.
    Instead of adding a position vector, it ROTATES the query/key vectors by
    an angle proportional to position. Generalizes better to longer sequences
    than learned absolute position embeddings."""
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        cos = self.cos[:seq_len].to(x.device)
        sin = self.sin[:seq_len].to(x.device)
        return (x * cos) + (self.rotate_half(x) * sin)


# ============================================================================
# TEMPLATE 1: GPT-STYLE DECODER-ONLY TRANSFORMER
# The workhorse architecture behind GPT-2/3/4, LLaMA, Mistral, etc.
# Good default choice for "generate text one token at a time."
# ============================================================================

class GPTStyleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        b, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))
        mask = causal_mask(seq_len, x.device)
        h = self.transformer(h, mask=mask)
        h = self.ln_f(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 2: LLAMA-STYLE TRANSFORMER
# Modern decoder-only design: RMSNorm + RoPE + SwiGLU instead of
# LayerNorm + learned positions + ReLU MLP. Closer to real production LLMs.
# ============================================================================

class LlamaStyleBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x, mask):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class LlamaStyleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(d_model // n_heads, max_len)
        self.blocks = nn.ModuleList([LlamaStyleBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        seq_len = x.size(1)
        h = self.token_emb(x)  # RoPE is normally applied inside attention on q/k;
                                # simplified here to keep the block swappable/readable.
        mask = causal_mask(seq_len, x.device)
        for block in self.blocks:
            h = block(h, mask)
        h = self.norm_f(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 3: ENCODER-ONLY TRANSFORMER (BERT-STYLE)
# Good for classification, embeddings, masked-language-modeling — NOT for
# free-form text generation (no causal mask, sees the whole sequence at once).
# ============================================================================

class BERTStyleEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)  # e.g. for masked-token prediction

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)  # no causal mask -> full bidirectional context
        return self.head(h)


# ============================================================================
# TEMPLATE 4: ENCODER-DECODER TRANSFORMER (T5 / translation-style)
# Good for sequence-to-sequence tasks: translation, summarization,
# question -> answer, source -> paraphrase, etc.
# ============================================================================

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4,
                 n_enc_layers=3, n_dec_layers=3, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_enc_layers, num_decoder_layers=n_dec_layers,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        src_pos = torch.arange(src_ids.size(1), device=src_ids.device)
        tgt_pos = torch.arange(tgt_ids.size(1), device=tgt_ids.device)
        src = self.token_emb(src_ids) + self.pos_emb(src_pos)
        tgt = self.token_emb(tgt_ids) + self.pos_emb(tgt_pos)
        tgt_mask = causal_mask(tgt_ids.size(1), tgt_ids.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.head(out)

    def forward_lm_style(self, x):
        """Compatibility shim so this can be dropped into a train.py written
        for decoder-only models: treats x as both src and tgt (autoencoding)."""
        return self.forward(x, x)


# ============================================================================
# TEMPLATE 5: MIXTURE-OF-EXPERTS (MoE) TRANSFORMER
# Used in Mixtral, GPT-4 (rumored), Switch Transformer. Instead of one big
# feed-forward block per layer, there are several "expert" feed-forward
# blocks, and a small router network picks which experts handle each token.
# Lets you scale total parameters way up without scaling compute-per-token
# by nearly as much — relevant if your research concerns scaling laws.
# ============================================================================

class MoEFeedForward(nn.Module):
    def __init__(self, d_model, n_experts=4, top_k=2, hidden_mult=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * hidden_mult),
                nn.GELU(),
                nn.Linear(d_model * hidden_mult, d_model),
            ) for _ in range(n_experts)
        ])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        b, s, d = x.shape
        flat = x.reshape(-1, d)                          # (b*s, d)
        router_logits = self.router(flat)                 # (b*s, n_experts)
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            gate = weights[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                token_mask = (expert_idx == e)
                if token_mask.any():
                    out[token_mask] += gate[token_mask] * self.experts[e](flat[token_mask])
        return out.reshape(b, s, d)


class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe_ff = MoEFeedForward(d_model, n_experts, top_k)

    def forward(self, x, mask):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.moe_ff(self.norm2(x))
        return x


class MoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 n_experts=4, top_k=2, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, n_experts, top_k) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = causal_mask(x.size(1), x.device)
        for block in self.blocks:
            h = block(h, mask)
        h = self.norm_f(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 6: SLIDING-WINDOW / LOCAL ATTENTION TRANSFORMER
# Used in Longformer, Mistral (as one component). Each token only attends to
# nearby tokens within a fixed window, instead of the whole sequence. Makes
# attention cost scale linearly instead of quadratically with sequence
# length — relevant if your research involves very long documents.
# ============================================================================

class LocalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, window=32, dropout=0.1):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def _local_mask(self, seq_len, device):
        # Combine causal masking with a local window: token i can only see
        # tokens in [i-window, i].
        idx = torch.arange(seq_len, device=device)
        dist = idx.unsqueeze(0) - idx.unsqueeze(1)  # (seq_len, seq_len), query - key
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill((dist < 0) | (dist > self.window), float("-inf"))
        return mask

    def forward(self, x):
        h = self.norm1(x)
        mask = self._local_mask(x.size(1), x.device)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class LocalAttentionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 window=32, max_len=2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            LocalAttentionBlock(d_model, n_heads, window) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 7: TRANSFORMER-XL STYLE (SEGMENT-LEVEL RECURRENCE)
# Carries a "memory" of hidden states from the previous chunk of text into
# the next chunk, letting the model have context longer than one forward
# pass could normally hold — relevant for research on long-range dependency.
# NOTE: this template's forward() has a different signature (returns memory
# too) since it's inherently stateful across calls.
# ============================================================================

class TransformerXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, memory=None):
        # memory: previous segment's hidden states, prepended as extra
        # keys/values so attention can "see" recent past beyond this segment.
        kv_input = x if memory is None else torch.cat([memory, x], dim=1)
        h = self.norm1(x)
        kv = self.norm1(kv_input)
        seq_len = x.size(1)
        mem_len = kv_input.size(1) - seq_len
        mask = torch.full((seq_len, kv_input.size(1)), float("-inf"), device=x.device)
        for i in range(seq_len):
            mask[i, :mem_len + i + 1] = 0  # can see all memory + up to itself
        attn_out, _ = self.attn(h, kv, kv, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerXLStyle(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_len=512, mem_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerXLBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.mem_len = mem_len

    def forward(self, x, memories=None):
        """Returns (logits, new_memories). Pass new_memories back in as
        `memories` on your NEXT call (e.g. next chunk of the same document)
        to give the model recurrence across chunks."""
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        new_memories = []
        for i, block in enumerate(self.blocks):
            mem = memories[i] if memories is not None else None
            h = block(h, memory=mem)
            new_memories.append(h[:, -self.mem_len:].detach())
        h = self.norm_f(h)
        return self.head(h), new_memories


# ============================================================================
# TEMPLATE 8: RWKV-STYLE (RNN / TRANSFORMER HYBRID)
# Real RWKV uses a custom WKV linear-attention recurrence; this is a
# SIMPLIFIED educational version capturing the core idea: replace quadratic
# self-attention with a linear recurrent update, giving RNN-like inference
# cost with transformer-like training parallelism ambitions.
# ============================================================================

class RWKVStyleBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, x):
        h = self.norm1(x)
        k, v, r = self.key(h), self.value(h), torch.sigmoid(self.receptance(h))
        decay = torch.sigmoid(self.time_decay)
        # Simple linear recurrence over the time dimension (sequential scan).
        b, s, d = k.shape
        state = torch.zeros(b, d, device=x.device)
        outs = []
        for t in range(s):
            state = decay * state + k[:, t] * v[:, t]
            outs.append(state)
        wkv = torch.stack(outs, dim=1)
        x = x + self.output(r * wkv)
        x = x + self.ff(self.norm2(x))
        return x


class RWKVStyleModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([RWKVStyleBlock(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x)  # no positional embedding needed -- recurrence
                                # encodes order implicitly, like an RNN
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 9: STATE-SPACE MODEL / MAMBA-STYLE (SIMPLIFIED)
# Real Mamba uses a hardware-optimized selective state-space scan; this is a
# SIMPLIFIED educational stand-in for research/teaching purposes, capturing
# the core idea of a per-channel linear recurrence with input-dependent
# gating instead of attention. Sub-quadratic cost, relevant for long-sequence
# research.
# ============================================================================

class SimplifiedSSMBlock(nn.Module):
    def __init__(self, d_model, state_size=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.A_log = nn.Parameter(torch.randn(d_model, state_size) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, state_size) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, state_size) * 0.01)
        self.out_proj = nn.Linear(d_model, d_model)
        self.state_size = state_size

    def forward(self, x):
        h_in = self.norm(x)
        gate_and_val = self.in_proj(h_in)
        val, gate = gate_and_val.chunk(2, dim=-1)
        val = val * torch.sigmoid(gate)  # input-dependent gating

        b, s, d = val.shape
        A = -torch.exp(self.A_log)  # (d_model, state_size), kept stable/negative
        state = torch.zeros(b, d, self.state_size, device=x.device)
        outs = []
        for t in range(s):
            u_t = val[:, t].unsqueeze(-1)              # (b, d, 1)
            state = state * torch.exp(A) + u_t * self.B  # recurrent state update
            y_t = (state * self.C).sum(-1)               # (b, d)
            outs.append(y_t)
        y = torch.stack(outs, dim=1)
        return x + self.out_proj(y)


class SimplifiedSSMModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, state_size=16, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SimplifiedSSMBlock(d_model, state_size) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 10: SIMPLE RNN (VANILLA RECURRENT NETWORK)
# The oldest/simplest sequence architecture. Mostly useful as a research
# baseline to show how much transformers/SSMs improve over plain recurrence.
# ============================================================================

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(d_model, d_model, num_layers=n_layers, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x)
        h, _ = self.rnn(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 11: LSTM
# Classic gated recurrent architecture, strong baseline before transformers.
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                             batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x)
        h, _ = self.lstm(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 12: GRU
# Similar to LSTM but with fewer gates/parameters — often trains faster with
# comparable performance, useful as a lightweight recurrent baseline.
# ============================================================================

class GRUModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x)
        h, _ = self.gru(h)
        return self.head(h)


# ============================================================================
# TEMPLATE 13: RETRIEVAL-AUGMENTED WRAPPER (RAG-STYLE SKELETON)
# Wraps ANY other template and lets you inject external "retrieved" context
# tokens (e.g. from a document database) alongside the normal input, before
# generation. This is a SKELETON — you supply the retrieval mechanism
# (nearest-neighbor search, a vector DB, etc.) yourself; this class just
# shows the plug point where retrieved context joins the model.
# ============================================================================

class RetrievalAugmentedWrapper(nn.Module):
    def __init__(self, base_model, d_model, max_retrieved_len=64):
        super().__init__()
        self.base_model = base_model          # any decoder-only template from above
        self.retrieved_proj = nn.Linear(d_model, d_model)
        self.max_retrieved_len = max_retrieved_len

    def forward(self, input_ids, retrieved_ids=None):
        """retrieved_ids: (batch, retrieved_len) token ids of externally
        fetched context (e.g. top-k similar Wikipedia passages). If provided,
        they are prepended to input_ids before the base model sees them.
        You are responsible for the retrieval step itself (e.g. FAISS,
        embedding similarity search) BEFORE calling this forward()."""
        if retrieved_ids is not None:
            combined = torch.cat([retrieved_ids, input_ids], dim=1)
        else:
            combined = input_ids
        return self.base_model(combined)


# ============================================================================
# TEMPLATE 14: HIERARCHICAL / CHUNKED TRANSFORMER
# Encodes long input in fixed-size chunks with a local transformer, then runs
# a second, smaller transformer OVER the chunk summaries — useful research
# direction for very long scientific documents (e.g. full papers, genomes).
# ============================================================================

class HierarchicalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4,
                 chunk_size=64, n_local_layers=2, n_global_layers=2, max_chunks=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.local_pos = nn.Embedding(chunk_size, d_model)
        local_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.local_transformer = nn.TransformerEncoder(local_layer, n_local_layers)

        self.chunk_pos = nn.Embedding(max_chunks, d_model)
        global_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.global_transformer = nn.TransformerEncoder(global_layer, n_global_layers)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, seq_len = x.shape
        pad = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad > 0:
            x = F.pad(x, (0, pad), value=0)
        n_chunks = x.size(1) // self.chunk_size

        h = self.token_emb(x)
        local_pos = torch.arange(self.chunk_size, device=x.device)
        h = h + self.local_pos(local_pos).repeat(n_chunks, 1)
        h = h.view(b * n_chunks, self.chunk_size, -1)
        h = self.local_transformer(h)                   # encode each chunk locally

        chunk_summaries = h.mean(dim=1).view(b, n_chunks, -1)  # summarize each chunk
        chunk_pos = torch.arange(n_chunks, device=x.device)
        chunk_summaries = chunk_summaries + self.chunk_pos(chunk_pos)
        chunk_summaries = self.global_transformer(chunk_summaries)  # relate chunks to each other

        # Broadcast global context back down and combine with local features
        global_ctx = chunk_summaries.unsqueeze(2).expand(-1, -1, self.chunk_size, -1)
        global_ctx = global_ctx.reshape(b * n_chunks, self.chunk_size, -1)
        h = h + global_ctx

        h = h.view(b, n_chunks * self.chunk_size, -1)[:, :seq_len]
        return self.head(h)


# ============================================================================
# REGISTRY: the "menu" train.py picks from.
# Add new architectures above, then register them here with a string key.
# ============================================================================

MODEL_TEMPLATES = {
    "gpt_style": GPTStyleTransformer,
    "llama_style": LlamaStyleTransformer,
    "bert_style": BERTStyleEncoder,
    "encoder_decoder": EncoderDecoderTransformer,
    "moe_transformer": MoETransformer,
    "local_attention": LocalAttentionTransformer,
    "transformer_xl": TransformerXLStyle,
    "rwkv_style": RWKVStyleModel,
    "ssm_style": SimplifiedSSMModel,
    "simple_rnn": SimpleRNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "hierarchical": HierarchicalTransformer,
    # "retrieval_wrapper" isn't listed directly since it WRAPS another
    # template rather than standing alone -- build it in train.py like:
    #   base = MODEL_TEMPLATES["gpt_style"](vocab_size, d_model=256)
    #   model = RetrievalAugmentedWrapper(base, d_model=256)
}
