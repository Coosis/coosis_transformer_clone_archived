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

if os.path.exists("word_level_transformer/woo.txt"):
    with open('word_level_transformer/woo.txt', 'r', encoding='utf-8') as f:
        text = f.read()
else:
    print("woo.txt not found. ")
    exit()

tokens = re.findall(r"\w+|\W+", text)
# here are all the unique words that occur in this text
words = sorted(list(set(tokens)))
vocab_size = len(words)
# create a mapping from words to integers
stoi = { w:i for i,w in enumerate(words) }
itos = { i:w for i,w in enumerate(words) }
encode = lambda s: stoi[s] # encoder: take a string, output an integer
decode = lambda l: itos[l] # decoder: take an integer, output a string

model = gpt(vocab_size, block_size, n_blocks, n_embd, n_head, head_size, dropout, device)
if not os.path.exists("word_level_transformer/model.pth"):
    exit()
else:
    model.load_state_dict(torch.load('word_level_transformer/model.pth'))
    print("model.pth loaded successfully. ")

if not os.path.exists("word_level_transformer/hyperparameters.txt"):
    print("hyperparameters.txt not found. ")
    exit()
else:
    with open("word_level_transformer/hyperparameters.txt", "r") as f:
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

encoded_data = [encode(word) for word in query.split()]

context = torch.tensor(encoded_data, dtype=torch.long, device=device)
context = context.unsqueeze(0)  # Add batch dimension
output = model.generate(context, max_lenth, block_size)[0]
decoded_output = ""
for i in output.tolist():
    decoded_output += decode(i)
print(decoded_output)
