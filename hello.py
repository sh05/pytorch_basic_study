import torch
print(torch.backends.mps.is_available())
device = torch.device('mps')
input = torch.tensor([1, 2, 3], dtype=torch.float)
# GPUのメモリに移動
input = input.to(device)
print(input)


