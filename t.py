import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np

w = torch.tensor([[1,2],[3,4],[5,6]], dtype=torch.float, requires_grad=True)

a = torch.ones(3,4)
print(w.shape, a.shape)
o = torch.matmul(w.T.T.T, a)
j = torch.sum(o)
print(o.shape)
torch.flip()
(j - 1).backward()

print(w.grad)

np.argsort