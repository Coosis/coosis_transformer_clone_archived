import os

new_token = 10

root_path = os.path.dirname(__file__)

def read_all(path):
    full_text = ""
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for file in txt_files:
        with open(path + file, 'r', encoding='utf-8') as f:
            text = f.read()
        full_text += text

    return full_text

data_path = f'{root_path}/dataset/'
text = read_all(data_path)

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
else:
    vocab = sorted(set(text))
    print("vocab.txt not found. Creating an empty one.")

vocab = sorted(vocab)
print("Original vocab size: ", len(vocab))
print(f"First 10 words: {vocab[:10]}")

occurrences = []
pairs = []
for word in vocab:
    occurrences.append(text.count(word))

for fir in vocab:
    for sec in vocab:
        if fir == sec:
            continue

        pair = fir + sec
        # print(pair)
        pairs.append(pair)
        occurrences.append(text.count(pair))

print(occurrences)
print(pairs)

for i in range(new_token):
    max_occ = max(occurrences)
    max_occ_index = occurrences.index(max_occ)
    print(f'Max occ: {max_occ}, index: {max_occ_index}')
    vocab.append(pairs[max_occ_index])
    print(f'New token: {pairs[max_occ_index]}')

vocab = sorted(vocab)
with open(f'{root_path}/vocab.txt', "w", encoding='utf-8') as f:
    for word in vocab:
        f.write(word + '\n')

print("New vocab size: ", len(vocab))
print(f'New tokens: {vocab[-new_token:]}')
print(f'Vocabulary built successfully.')