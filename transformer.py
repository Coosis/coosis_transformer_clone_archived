import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
n_embd = 256
n_head = 16
n_blocks = 8
head_size = 32
batch_size = 16
block_size = 64
dropout = 0.1
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('woo.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Single Attention Head
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, block_size, n_embd)
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

# Multi Attention Head
class MultiattentionHead(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, block_size, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # (batch_size, block_size, n_embd)
        return self.net(x)

# Transformer Block
class Block(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.sa = MultiattentionHead(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # (batch_size, block_size, n_embd)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Transformer
class GPTLanguageModel(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.embd_table = nn.Embedding(vocab_size, n_embd)
        self.posit_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_size, dropout) for _ in range(n_blocks)])
        self.lnf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, Target=None):
        # (batch_size, block_size)
        B, T = idx.shape
        # (batch_size, block_size, n_embd)
        x = self.embd_table(idx) + self.posit_table(torch.arange(T, device=device))
        # (batch_size, block_size, n_embd)
        x = self.blocks(x)
        # (batch_size, block_size, n_embd)
        x = self.lnf(x)
        # (batch_size, block_size, vocab_size)
        logits = self.lm_head(x)

        if Target == None:
            loss = 0
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            Target = Target.view(B*T)
            loss = F.cross_entropy(logits, Target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_con = idx[:, -block_size:]
            logits, loss = self(idx_con)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = GPTLanguageModel(n_embd, n_head, head_size, dropout)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 1000)[0].tolist()))