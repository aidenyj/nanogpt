import torch
import math
import random

device = "mps"

batch_size = 64
block_size = 256

f = open("input.txt", "r").read()

chars = sorted(list(set(f)))

chartonum = {c : i for i,c in enumerate(chars)}
numtochar = {i : c for i,c in enumerate(chars)}

encode = lambda s: [chartonum[t] for t in s]
decode = lambda x: [numtochar[y] for y in x]

data = torch.tensor(encode(f))
train = data[:math.floor(len(data)*0.9)]
val = data[math.floor(len(data)*0.9):]

def get_batch(where):
    x = torch.empty(batch_size,block_size)
    y = torch.empty(batch_size,block_size)
    for k in range(batch_size):
        i = random.randint(0,len(where)-block_size-1)
        x[k]=where[i:i+block_size]
        y[k]=where[i+1:i+block_size+1]
    return x.to(device), y.to(device)
