import torch
import torch.nn as nn 
from torch.nn import functional as F

import requests

# Save the link to the data, to later be able to delete it from the directory to save storage
link_to_data = f"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" 
filename = "tiny_shakespeare.txt"

# HYPERPARAMETERS
batch_size = 32
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cpu"
eval_iters = 200
n_embed = 32

torch.manual_seed(1337)

# GET DATA
def download_data(link, filename):
    r = requests.get(link_to_data)

    with open(filename, "w") as f:
        f.write(r.text)
# download_data(link=link_to_data, filename=filename)

# LOAD DATA
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# CREATE ENCODING AND DECODING FUNCTIONS
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# CREATE TRAIN AND VAL SPLITS
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# FUNCTION TO GET A BATCH OF DATA
def get_batch(split: str = "train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# FUNCTIONS TO ESTIMATE LOSS
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            # Get a batch of data 
            X, y = get_batch(split)
            # Make predictions
            logits, loss = model(X, y)
            losses[i] = loss.item()

        out[split] = losses.mean()  
    model.train()
    return out

# CREATE A SIMPLE BASELINE BIGRAM MODEL
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel().to(device)

# CREATE A PYTORCH OPTMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# TRAIN THE MODEL
for i in range(max_iters):

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch: {i}/{max_iters}: train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch()

    # Make predictions
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# GENERATE TEXT FROM THE MODEL
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))



