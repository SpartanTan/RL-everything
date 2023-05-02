import torch
import math
torch.linspace(-math.pi, math.pi, 2000)
x = torch.linspace(-math.pi, math.pi, 2000)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)