
import torch
import torch.nn as nn

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):  
        x = self.fc1(x)
        print("type(fc1_output):", x.dtype)
        x = self.relu(x)
        x = self.ln(x) 
        print("type(ln_output):", x.dtype)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = ToyModel(10, 10)
    model.to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())

    x = torch.randn(10, 10).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        print("type(model_params):", model.fc1.weight.dtype)
        y = model(x)
        print("type(logits):", y.dtype)
        loss = torch.nn.functional.cross_entropy(y, x)
        print("type(loss):", loss.dtype)
        optimizer.zero_grad()
        loss.backward()
        print("type(grads):", model.fc1.weight.grad.dtype)
        optimizer.step()