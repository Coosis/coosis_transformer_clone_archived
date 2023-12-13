import os.path
import re # regular expressions

import tiktoken

from wlt import GPTLanguageModel as gpt
from wlt import estimate_loss, get_batch
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

def read_all(path):
    text = ""
    data_length = []
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for file in txt_files:
        prev_len = len(text)
        with open(path + file, 'r', encoding='utf-8') as f:
            text += f.read()
        data_length.append(len(text) - prev_len)
    return text, data_length

data_path = "word_level_transformer/dataset/"
text, data_length = read_all(data_path)

if text == "":
    print("No dataset found. ")
    exit()

parameters_path = "word_level_transformer/hyperparameters.txt"
if not os.path.exists(parameters_path):
    print("hyperparameters.txt not found. Using Default parameters.")
    with open(parameters_path, "w") as f:
        toWrite = ""
        toWrite += f"{n_embd}\n"
        toWrite += f"{n_head}\n"
        toWrite += f"{n_blocks}\n"
        toWrite += f"{head_size}\n"
        toWrite += f"{batch_size}\n"
        toWrite += f"{block_size}\n"
        toWrite += f"{dropout}\n"
        toWrite += f"{max_iters}\n"
        toWrite += f"{eval_interval}\n"
        toWrite += f"{learning_rate}\n"
        toWrite += f"{eval_iters}\n"
        f.write(toWrite)
else:
    with open(parameters_path, "r") as f:
        lines = f.readlines()
        n_embd = int(lines[0])
        n_head = int(lines[1])
        n_blocks = int(lines[2])
        head_size = int(lines[3])
        batch_size = int(lines[4])
        block_size = int(lines[5])
        dropout = float(lines[6])
        max_iters = int(lines[7])
        eval_interval = int(lines[8])
        learning_rate = float(lines[9])
        eval_iters = int(lines[10])
    print("hyperparameters.txt loaded successfully. ")

# tokens = re.findall(r"\w+|\W+", text)
# here are all the unique words that occur in this text
# words = sorted(list(set(tokens)))

# create a mapping from words to integers
# stoi = { w:i for i,w in enumerate(words) }
# itos = { i:w for i,w in enumerate(words) }
# encode = lambda s: stoi[s] # encoder: take a string, output an integer
# decode = lambda l: itos[l] # decoder: take an integer, output a string

enc = tiktoken.get_encoding("cl100k_base")
encode = enc.encode
decode = enc.decode

# Train and test splits
# encoded_data = [encode(word) for word in tokens]
encoded_data = encode(text)
vocab_size = max(set(encoded_data))
print(f"Vocab size: {vocab_size}")
data = torch.tensor(encoded_data, dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
for l in leng

model = gpt(vocab_size, block_size, n_blocks, n_embd, n_head, head_size, dropout, device)
model_path = "word_level_transformer/model.pth"
if not os.path.exists(model_path):
    print("model.pth not found. Using default parameters. ")
else:
    model.load_state_dict(torch.load(model_path))
    print("model.pth loaded successfully. ")

m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

state_dict = model.state_dict()
torch.save(state_dict, model_path)

print("Training ended successfully.")