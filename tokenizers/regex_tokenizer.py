### BPE tokenizer but using GPT4 Regex pattern
import regex
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
pat = regex.compile(GPT4_SPLIT_PATTERN)

def get_stats(ids):
    counts = {}
    for chunk in ids:
        for i in range(len(chunk) - 1):
            pair = (chunk[i], chunk[i+1])
            if pair in counts:
                counts[pair] += 1
            else:
                counts[pair] = 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    for chunk in ids:
        new_chunk = []
        i = 0
        while i < len(chunk):
            if i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i+1] == pair[1]:
                new_chunk.append(idx)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        new_ids.append(new_chunk)
    return new_ids

def train(text, vocab_size, pat):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    chunks = pat.findall(text)
    ids = [list(chunk.encode("utf-8")) for chunk in chunks]
    merges = {}
    num_merges = vocab_size - 256
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break

        pair = max(stats, key=stats.get)
        idx = 256 + i

        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge(ids, pair, idx)
    return merges, vocab

def encode(text, merges, pat):
    chunks = pat.findall(text)
    ids = [list(chunk.encode("utf-8")) for chunk in chunks]
    while True:
        stats = get_stats(ids)
        if not stats:
            break
            
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        
        if pair not in merges:
            break
            
        ids = merge(ids, pair, merges[pair])
    merged_ids = [id for chunk in ids for id in chunk]
    return merged_ids

def decode(ids, vocab):
    tokens = []
    for id in ids:
        if id in vocab:
            tokens.append(vocab[id])
        else:
            tokens.append(bytes([id]))
    tokens = b"".join(tokens)
    return tokens.decode("utf-8", errors="replace")

text = "Idiot! Idly?"
merges, vocab = train(text, 256 + 10, pat)
print(encode(text, merges, pat))
print(decode(encode(text, merges, pat), vocab))
