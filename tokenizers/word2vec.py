import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

def build_vocab(sentences: List[str]) -> Dict[str, int]:
    """
    Builds a dictionary mapping words to integer indices.
    """
    vocab = set()
    for sentence in sentences:
        words = sentence.lower().strip().split()
        vocab.update(words)

    return {word: i for i, word in enumerate(sorted(vocab))}

def text_to_indices(sentences: List[str], word_to_int: Dict[str, int]) -> List[List[int]]:
    """
    Converts sentences of text into lists of integer indices.
    """
    corpus_indices = []
    for sentence in sentences:
        words = sentence.lower().strip().split()
        sentence_indices = [word_to_int[word] for word in words if word in word_to_int]
        corpus_indices.append(sentence_indices)
                   
    return corpus_indices    

def generate_skipgram_data(corpus_indices: List[List[int]], window_size: int = 2) -> List[Tuple[int, int]]:
    """
    Generates (center_word, context_word) pairs for Skip-gram training.
    """
    training_data = []
    
    for sentence in corpus_indices:
        sentence_length = len(sentence)
        for center_pos, center_word_id in enumerate(sentence):
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                
                context_pos = center_pos + offset
                if 0 <= context_pos < sentence_length:
                    context_word_id = sentence[context_pos]
                    training_data.append((center_word_id, context_word_id))
    
    return training_data

class Word2VecDataset(Dataset):
    def __init__(self, training_data: List[Tuple[int, int]]):
        self.data = training_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target, context = self.data[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, x):
        embeds = self.embedding(x)  # Shape: (Batch_Size, Embedding_Dim)
        out = self.linear(embeds)   # Shape: (Batch_Size, Vocab_Size)
        return out
    

EMBEDDING_DIM = 10
WINDOW_SIZE = 2
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.01
    
with open("small_text.txt", "r") as f:
    raw_sentences = f.readlines()
word_to_int = build_vocab(raw_sentences)
print(word_to_int)
corpus_indices = text_to_indices(raw_sentences, word_to_int)
print(corpus_indices)
training_data = generate_skipgram_data(corpus_indices, window_size=WINDOW_SIZE)
print(training_data)

dataset = Word2VecDataset(training_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vocab_size = len(word_to_int)
model = Word2VecModel(vocab_size, EMBEDDING_DIM)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    total_loss = 0
    for target_batch, context_batch in dataloader:
        # 1. Forward Pass
        preds = model(target_batch)
        
        # 2. Compute Loss
        loss = loss_fn(preds, context_batch)
        
        # 3. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        
        
embeddings = model.embedding.weight.detach().numpy()

idx = word_to_int['click']
vector = embeddings[idx]

def get_similarity(word1, word2):
    v1 = embeddings[word_to_int[word1]]
    v2 = embeddings[word_to_int[word2]]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))