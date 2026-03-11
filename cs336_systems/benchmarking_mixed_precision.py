import torch
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        print(f"fc1 dtype: {x.dtype}")
        x = self.relu(x)
        print(f"relu dtype: {x.dtype}")
        x = self.ln(x)
        print(f"ln dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"fc2 dtype: {x.dtype}")
        return x

model = ToyModel(10, 10).to('cuda')
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss = CrossEntropyLoss()

x = torch.randn(10, 10).to('cuda')
gt = torch.randn(10, 10).to('cuda')

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    print(f"parameters dtype: {next(model.parameters()).dtype}")
    y = model(x)
    print(f"logits dtype: {y.dtype}")
    train_loss = loss(y, gt)
    print(f"train loss dtype: {train_loss.dtype}")
    train_loss.backward()
    print(f"gradients dtype: {next(model.parameters()).grad.dtype}")
    optimizer.step()