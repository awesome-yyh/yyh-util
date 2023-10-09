import torch
import torch.nn as nn


loss_fn = nn.CrossEntropyLoss()
model = torch.nn.Linear(10, 2)

print("=== SGD ===")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(inputs), targets).backward()
optimizer.step()

print("=== SGDM ===")
# torch.optim.SGDM

print("=== AdaGrad ===")
torch.optim.Adagrad

print("=== RMSProp ===")
torch.optim.RMSprop

print("=== Adam ===")
torch.optim.Adam

print("=== AdamW ===")
torch.optim.AdamW
