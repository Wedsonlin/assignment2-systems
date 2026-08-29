import argparse
import timeit
from statistics import fmean, pstdev
from typing import Literal

import torch
import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def benchmark(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_steps: int = 5,
    execution_steps: int = 10,
    pattern: Literal["forward-only", "forward-and-backward", "full-training-step"] = "forward-only",
    autocast: bool = True
):
    batch_size = 4
    vocab_size = model.config["vocab_size"]
    context_length = model.config["context_length"]

    def make_batches(steps):
        shape = (steps, batch_size, context_length)
        return (
            torch.randint(vocab_size, shape, device=device),
            torch.randint(vocab_size, shape, device=device),
        )

    warmup_inputs, warmup_targets = make_batches(warmup_steps)
    execution_inputs, execution_targets = make_batches(execution_steps)

    if pattern not in {"forward-only", "forward-and-backward", "full-training-step"}:
        raise ValueError(f"Unsupported benchmark pattern: {pattern}")

    def step(inputs, targets):
        if pattern == "forward-only":
            model(inputs)
            return

        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        if pattern == "full-training-step":
            optimizer.step()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
        for inputs, targets in zip(warmup_inputs, warmup_targets):
            step(inputs, targets)
        torch.cuda.synchronize()

        start = timeit.default_timer()
        for inputs, targets in zip(execution_inputs, execution_targets):
            step(inputs, targets)
            torch.cuda.synchronize()
        return (timeit.default_timer() - start) / execution_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--pattern", type=str, default="forward-only")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--execution-steps", type=int, default=10)
    parser.add_argument("--repeat-times", type=int, default=10)
    parser.add_argument("--autocast", type=str, default="true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )
    model.to(device)
    optimizer = AdamW(model.parameters())
    
    latencies = [
        benchmark(
            model=model, 
            optimizer=optimizer,
            device=device,
            warmup_steps=args.warmup_steps,
            execution_steps=args.execution_steps,
            pattern=args.pattern,
            autocast=True if args.autocast == "true" else False,
        )
        for _ in range(args.repeat_times)
    ]
    avg = fmean(latencies)
    std = pstdev(latencies)
    print(f"Latency: {avg:.5f} seconds ± {std:.5f} seconds")
