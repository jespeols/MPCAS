# %% To understand torch.all() and how validation is done in PyTorch
import torch

x = torch.randint(0, 2, (5, 2))
y = torch.randint(0, 2, (5, 2))

print(x)
print(y)

print(x == y)
print(torch.all(x == y, dim = 1))
print(torch.all(x == y, dim = 1).sum())
