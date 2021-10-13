import torch
import torch.nn as nn

a = nn.Sigmoid()
space = torch.linspace(-5, 5, steps=11)[None, :]
tensor = space.repeat(2, 1)

BCE = nn.BCELoss()
print(BCE(a(tensor), torch.ones_like(tensor)))