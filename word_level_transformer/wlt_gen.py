import os
import re

import tiktoken

from wlt import GPTLanguageModel as gpt
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
max_iters = 400
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_lenth = 100

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

enc = tiktoken.get_encoding("cl100k_base")
encoded_data = enc.encode(text)
tokens = sorted(list(set(encoded_data)))
ttoi = { t:i for i,t in enumerate(tokens) }
itot = { i:t for i,t in enumerate(tokens) }
encode = lambda s: [ttoi[t] for t in enc.encode(s)]
decode = lambda l: enc.decode([itot[i] for i in l])

vocab_size = len(tokens)
print(f"Vocab size: {vocab_size}")

model = gpt(vocab_size, block_size, n_blocks, n_embd, n_head, head_size, dropout, device)

model_path = "word_level_transformer/model.pth"
if not os.path.exists(model_path):
    exit()
else:
    model.load_state_dict(torch.load(model_path))
    print("model.pth loaded successfully. ")

parameters_path = "word_level_transformer/hyperparameters.txt"
if not os.path.exists(parameters_path):
    print("hyperparameters.txt not found. ")
    exit()
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

m = model.to(device)

query = input("Enter the query: ")

encoded_data = encode(query)

context = torch.tensor(encoded_data, dtype=torch.long, device=device)
context = context.unsqueeze(0)  # Add batch dimension
output = model.generate(context, max_lenth, block_size)[0]
decoded_output = decode(output.tolist())
print(decoded_output)
