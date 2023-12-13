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
    full_text = ""
    training_text = ""
    validation_text = ""
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for file in txt_files:
        with open(path + file, 'r', encoding='utf-8') as f:
            text = f.read()
            n = int(0.9*len(text)) # first 90% will be train, rest val
            training_text += text[:n]
            validation_text += text[n:]
        full_text += text

    return full_text, training_text, validation_text

data_path = "word_level_transformer/dataset/"
text, training_text, validation_text = read_all(data_path)

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
    
enc = tiktoken.get_encoding("cl100k_base")
encoded_data = enc.encode(text)
tokens = sorted(list(set(encoded_data)))
ttoi = { t:i for i,t in enumerate(tokens) }
itot = { i:t for i,t in enumerate(tokens) }
encode = lambda s: [ttoi[t] for t in enc.encode(s)]
decode = lambda l: enc.decode([itot[i] for i in l])

# Train and test splits
encoded_data = encode(text)
vocab_size = len(tokens)
print(f"Vocab size: {vocab_size}")
# data = torch.tensor(encoded_data, dtype=torch.long)

train_data = torch.tensor(encode(training_text), dtype=torch.long)
# print(training_text[:100])
val_data = torch.tensor(encode(validation_text), dtype=torch.long)

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