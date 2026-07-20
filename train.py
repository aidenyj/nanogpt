import torch
import torch.nn as nn
import math

device = "mps"

batch_size = 64
block_size = 256 # T
n_embed = 384
p = 0.2
num_heads = 6

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
        i = torch.randint(0,len(where)-block_size)
        x[k]=where[i:i+block_size]
        y[k]=where[i+1:i+block_size+1]
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size): # d_k 
        super().__init__()
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x): # B, T, n_embed
        k = self.key(x) # B, T, hs
        q = self.query(x)
        v = self.value(x)

        att = q @ torch.transpose(k, 1, 2)/math.sqrt(k.size(-1)) # B, T, T

        mask = -float('inf')*torch.ones(k.size(-2),k.size(-2), device = device)
        mask = torch.tril(mask,diagonal=-1).transpose(-1,-2)

        att = torch.softmax(att + mask, dim = -1)
        att = self.dropout(att)

        return att @ v # B, T, hs
    
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.w = nn.Linear(num_heads*head_size, n_embed)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        after_heads = torch.concat([H(x) for H in self.heads], dim = -1) # B, T, hx*nh
        
        resize = self.w(after_heads) # B, T, n_embed
        resize = self.dropout(resize)

        return resize
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(p)
        )
    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHead(num_heads=num_heads, head_size = n_embed // num_heads)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

