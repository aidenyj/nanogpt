import torch
from train import model, device, val, block_size

model.to(device)
model.load_state_dict(torch.load('ckpt.pt', map_location=device)['model'])

print(model.generate(val[:block_size].to(device), 1000))
