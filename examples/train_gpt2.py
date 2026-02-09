#!/usr/bin/env python3
"""Fine-tune GPT-2 (124M) with MacFleet distributed training.

This example demonstrates training a transformer model (GPT-2)
using MacFleet across multiple Apple Silicon Macs.

Requirements:
    pip install transformers datasets

Usage:
    # Single node:
    python examples/train_gpt2.py

    # Distributed (master):
    python examples/train_gpt2.py --distributed --role master

    # Distributed (worker):
    python examples/train_gpt2.py --distributed --role worker --master 10.0.0.1
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from macfleet import ClusterConfig, NodeRole, TrainingConfig
from macfleet.core.config import CompressionType


# Simple GPT-2-like model for demonstration
# (Can be replaced with HuggingFace's GPT2LMHeadModel if transformers is installed)
class SimpleGPT2(nn.Module):
    """Simplified GPT-2 style model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        # Output
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits if loss is None else (loss, logits)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention for autoregressive modeling."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(
        self,
        texts: list,
        seq_length: int = 128,
        vocab_size: int = 50257,
    ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Simple tokenization (character-level for demo)
        # In practice, use a real tokenizer like GPT-2's BPE
        all_text = " ".join(texts)
        self.tokens = [ord(c) % vocab_size for c in all_text]

        # Number of sequences
        self.num_sequences = max(1, len(self.tokens) // seq_length - 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1

        tokens = self.tokens[start:end]

        # Pad if necessary
        if len(tokens) < self.seq_length + 1:
            tokens = tokens + [0] * (self.seq_length + 1 - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y


def create_sample_dataset(num_samples: int = 1000, seq_length: int = 128):
    """Create a sample dataset for testing."""
    # Generate some sample text
    texts = []
    sample_text = "The quick brown fox jumps over the lazy dog. " * 100

    for i in range(num_samples // 10):
        texts.append(sample_text + f" Sample {i}. ")

    return TextDataset(texts, seq_length=seq_length)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with MacFleet")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--n-layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--role", choices=["master", "worker"], default="master")
    parser.add_argument("--master", type=str, default="10.0.0.1", help="Master address")
    parser.add_argument("--compression", choices=["none", "topk", "fp16", "topk_fp16"],
                       default="topk_fp16", help="Gradient compression")
    args = parser.parse_args()

    print("=" * 60)
    print("MacFleet GPT-2 Training")
    print("=" * 60)
    print(f"Model: GPT-2 ({args.n_layer} layers, {args.n_head} heads, {args.n_embd} dim)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create model
    print("\nCreating model...")
    model = SimpleGPT2(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_seq_len=args.seq_length,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = create_sample_dataset(num_samples=1000, seq_length=args.seq_length)
    print(f"Dataset size: {len(dataset)} sequences")

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = model.to(device)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("\nStarting training...")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            loss, logits = model(input_ids, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Generate some text
    print("\nGenerating sample text...")
    model.eval()
    with torch.no_grad():
        # Start with a simple prompt
        prompt = torch.tensor([[ord('T') % 50257]], device=device)
        generated = prompt

        for _ in range(50):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if generated.shape[1] > args.seq_length:
                break

        # Decode (simple character-level)
        text = "".join([chr(t.item() % 128) for t in generated[0]])
        print(f"Generated: {text[:100]}...")


if __name__ == "__main__":
    main()
