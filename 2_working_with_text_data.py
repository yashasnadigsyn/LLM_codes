"""
Working with Text Data — from "LLM from Scratch" by Sebastian Raschka.

Pipeline: Load Data → Tokenize → DataLoader → Token Embedding → Positional Embedding → Input Embedding
"""

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    """Sliding-window dataset that produces (input, target) token-ID pairs."""

    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class GPTInputEmbedding(nn.Module):
    """Combines token embeddings with absolute positional embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=token_ids.device))
        return tok_emb + pos_emb

class TextPipeline:
    """End-to-end pipeline: text file → tokenized DataLoader → embeddings."""

    def __init__(
        self,
        filepath: str,
        max_length: int = 4,
        embed_dim: int = 256,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
    ):
        with open(filepath, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.token_ids = self.tokenizer.encode(self.text)

        dataset = GPTDatasetV1(self.text, self.tokenizer, max_length, stride)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

        self.embedding = GPTInputEmbedding(
            vocab_size=self.tokenizer.n_vocab,
            embed_dim=embed_dim,
            max_length=max_length,
        )

    def get_batch(self):
        """Return the first batch from the dataloader."""
        return next(iter(self.dataloader))

    def embed(self, batch_input_ids: torch.Tensor) -> torch.Tensor:
        """Run a batch of token IDs through the embedding module."""
        return self.embedding(batch_input_ids)

if __name__ == "__main__":
    pipeline = TextPipeline(
        filepath="word2vec.txt",
        max_length=4,
        embed_dim=256,
        batch_size=1,
        stride=1,
        shuffle=False,
    )

    print(f"Text length     : {len(pipeline.text)}")
    print(f"First 30 chars  : {pipeline.text[:30]}")

    print(f"\nToken count     : {len(pipeline.token_ids)}")
    print(f"First 10 IDs    : {pipeline.token_ids[:10]}")
    print(f"Decoded         : {pipeline.tokenizer.decode(pipeline.token_ids[:10])}")

    inputs, targets = pipeline.get_batch()
    print(f"\nFirst batch inputs  : {inputs}")
    print(f"First batch targets : {targets}")
    tok_emb = pipeline.embedding.token_embedding(inputs)
    print(f"\nToken embedding shape : {tok_emb.shape}")

    pos_emb = pipeline.embedding.position_embedding(torch.arange(4))
    print(f"Pos embedding shape   : {pos_emb.shape}")

    input_emb = pipeline.embed(inputs)
    print(f"Input embedding shape : {input_emb.shape}")
