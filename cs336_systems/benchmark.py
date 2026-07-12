import timeit
import argparse
from typing import Literal

import numpy as np
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
    warmup_inputs = torch.randint(
            low=0,
            high=model.config["vocab_size"],
            size=(warmup_steps, batch_size, model.config["context_length"]),
            dtype=torch.long,
            device=device
        )
    warmup_targets = torch.randint(
            low=0,
            high=model.config["vocab_size"],
            size=(warmup_steps, batch_size, model.config["context_length"]),
            dtype=torch.long,
            device=device       
        )
    execution_inputs = torch.randint(
            low=0,
            high=model.config["vocab_size"],
            size=(execution_steps, batch_size, model.config["context_length"]),
            dtype=torch.long,
            device=device
        )
    execution_targets = torch.randint(
            low=0,
            high=model.config["vocab_size"],
            size=(execution_steps, batch_size, model.config["context_length"]),
            dtype=torch.long,
            device=device        
        )   

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
        if pattern == "forward-only":
            for i in range(warmup_steps):
                model(warmup_inputs[i])
            torch.cuda.synchronize()
            
            start = timeit.default_timer()
            for i in range(execution_steps):
                model(execution_inputs[i])
                torch.cuda.synchronize()
            end = timeit.default_timer()
            return (end - start) / execution_steps

        elif pattern == "forward-and-backward":
            for i in range(warmup_steps):
                optimizer.zero_grad()
                output = model(warmup_inputs[i])
                loss = cross_entropy(output, warmup_targets[i])
                loss.backward()
            torch.cuda.synchronize()

            start = timeit.default_timer()
            for i in range(execution_steps):
                optimizer.zero_grad()
                output = model(execution_inputs[i])
                loss = cross_entropy(output, execution_targets[i])
                loss.backward()
                torch.cuda.synchronize()
            end = timeit.default_timer()
            return (end - start) / execution_steps

        elif pattern == "full-training-step":
            for i in range(warmup_steps):
                optimizer.zero_grad()
                output = model(warmup_inputs[i])
                loss = cross_entropy(output, warmup_targets[i])
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            
            start = timeit.default_timer()
            for i in range(execution_steps):
                optimizer.zero_grad()
                output = model(execution_inputs[i])
                loss = cross_entropy(output, execution_targets[i])
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
            end = timeit.default_timer()
            return (end - start) / execution_steps


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
    
    lantecys = []
    for _ in range(args.repeat_times):
        lantecys.append(benchmark(
            model=model, 
            optimizer=optimizer,
            device=device,
            warmup_steps=args.warmup_steps,
            execution_steps=args.execution_steps,
            pattern=args.pattern,
            autocast=True if args.autocast == "true" else False,
        ))
    avg = np.mean(lantecys)
    std = np.std(lantecys)
    print(f"Latency: {avg:.5f} seconds ± {std:.5f} seconds")