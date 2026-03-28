import torch
import fast_walsh_hadamard_transform as fwht
x = torch.randn(1, 256).cuda()
y = fwht.fast_walsh_hadamard_transform(x)
print((y**2).sum() / (x**2).sum())
