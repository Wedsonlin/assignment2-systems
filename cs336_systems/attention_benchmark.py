import gc
import timeit
import argparse
from itertools import product
from typing import Callable

import numpy as np
import torch
from einops import rearrange

from cs336_basics.model import scaled_dot_product_attention

def benchmark_scaled_dot_product_attention(
    d_model: int,
    seq_len: int,
    batch_size: int = 8,
    warmup_steps: int = 10,
    execution_steps: int = 100,
    attention_func: Callable = scaled_dot_product_attention,
) -> tuple[np.float64, np.float64, np.float64]:
    torch.set_float32_matmul_precision("high")

    iota = torch.arange(seq_len, device="cuda")
    qi = rearrange(iota, "query -> query 1")
    kj = rearrange(iota, "key   -> 1   key")
    causal_mask = qi >= kj  # (query, key)
    
    for _ in range(warmup_steps):
        Q = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        attn_output = attention_func(Q, K, V, mask=causal_mask)
        loss = attn_output.sum()
        loss.backward()
        torch.cuda.synchronize()
        del attn_output, loss, Q, K, V

    forward_times = []
    backward_times = []
    before_backward_memories = []
    for _ in range(execution_steps):
        Q = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)

        torch.cuda.synchronize()
        forward_start = timeit.default_timer()
        attn_output = attention_func(Q, K, V, mask=causal_mask)
        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        forward_times.append(forward_end - forward_start)



        loss = attn_output.sum()
        torch.cuda.synchronize()
        before_backward_memory = torch.cuda.memory_allocated(device="cuda")
        before_backward_memories.append(before_backward_memory)
        
        backward_start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = timeit.default_timer()
        backward_times.append(backward_end - backward_start)

        del attn_output, loss, Q, K, V

    forward_times = np.array(forward_times)
    backward_times = np.array(backward_times)
    before_backward_memories = np.array(before_backward_memories)
    return forward_times.mean(), backward_times.mean(),before_backward_memories.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()

    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    if args.compiled:
        attention_func = torch.compile(scaled_dot_product_attention)
    else:
        attention_func = scaled_dot_product_attention

    for d_model, seq_len in product(d_models, seq_lens):
        try:
            forward_time, backward_time, before_backward_memory = benchmark_scaled_dot_product_attention(d_model, seq_len, attention_func=attention_func)
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory error for d_model: {d_model}, seq_len: {seq_len}")
        else:
            print(f"d_model: {d_model}, seq_len: {seq_len}, forward_time: {forward_time*1000:.3f} ms, backward_time: {backward_time*1000:.3f} ms, before_backward_memory: {before_backward_memory / 1024 / 1024:.3f} MiB")
    
        gc.collect()
        torch.cuda.empty_cache()