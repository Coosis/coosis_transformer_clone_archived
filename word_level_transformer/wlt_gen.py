import os
import re

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

root_path = os.path.dirname(__file__)  # Get the directory of the current file
def load_vocab(path):
    path = f'{root_path}/{path}'
    vocab = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # print(f'here{line}!')
            if line[-1] == '\n' and len(line) > 1:
                vocab.append(line[:-1])
            else:
                vocab.append(line)
    return vocab

if(os.path.exists(f'{root_path}/vocab.txt')):
    vocab = load_vocab('vocab.txt')
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
else:
    print("vocab.txt not found. Run build_vocab.py first. ")
    exit()

vtoi = { v:i for i,v in enumerate(vocab) }
itov = { i:t for i,t in enumerate(vocab) }
def encode(s): 
    encoded = []
    while len(s) > 0:
        for v in reversed(vocab):
            if s.startswith(v):
                encoded.append(vtoi[v])
                s = s[len(v):]
    return encoded
def decode(l):
    return ''.join([itov[i] for i in l])

parameters_path = f"{root_path}/hyperparameters.txt"
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

model = gpt(vocab_size, block_size, n_blocks, n_embd, n_head, head_size, dropout, device)

model_path = f'{root_path}/model.pth'
if not os.path.exists(model_path):
    print("model.pth not found. ")
    exit()
else:
    model.load_state_dict(torch.load(model_path))
    print("model.pth loaded successfully. ")

m = model.to(device)

query = input("Enter the query: ")

encoded_data = encode(query)
context = torch.tensor(encoded_data, dtype=torch.long, device=device)
context = context.unsqueeze(0)  # Add batch dimension
output = model.generate(context, max_lenth, block_size)[0]
decoded_output = decode(output.tolist())
print(decoded_output)
