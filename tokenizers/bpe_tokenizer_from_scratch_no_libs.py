### BPE Tokenizer from scratch without using any libraries like Counter, etc..abs

def get_stats(ids):
    counts = {}
    for id in range(len(ids) - 1):
        pair = (ids[id], ids[id+1])
        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def train(text, vocab_size, verbose=False):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    ids = list(text.encode('utf-8'))
    merges = {}
    num_merges = vocab_size - 256

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break

        pair = max(stats, key=stats.get)
        idx = 256 + i
        
        if verbose:
            print(f"Merging {pair} into {idx} (count: {stats[pair]})")
            
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge(ids, pair, idx)
        
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        
        if pair not in merges:
            break
            
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    tokens = []
    for id in ids:
        if id in vocab:
            tokens.append(vocab[id])
        else:
            tokens.append(bytes([id]))
    tokens = b"".join(tokens)
    return tokens.decode("utf-8", errors="replace")

def print_token_id_text(vocab, text):
    ids = list(text.encode("utf-8"))
    for id in ids:
        print(f"{id}: {vocab[id]}")

text = open("small_text.txt", "r").read()
vocab_size = 256
merges, vocab = train(text, vocab_size)
encoded = encode(text, merges)
decoded = decode(encoded, vocab)
print(f"Original: {text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")