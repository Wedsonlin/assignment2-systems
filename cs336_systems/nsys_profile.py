import math
import argparse
from typing import Literal

import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from torch import Tensor
from jaxtyping import Bool, Float
from einops import einsum, rearrange

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.nn_utils import softmax

@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    torch.cuda.synchronize()
        
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    torch.cuda.synchronize()

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    torch.cuda.synchronize()

    return output

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def profile(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_steps: int = 5,
    execution_steps: int = 10
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


    for i in range(warmup_steps):
        optimizer.zero_grad()
        output = model(warmup_inputs[i])
        loss = cross_entropy(output, warmup_targets[i])
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    with nvtx.range("benchmark_execution"):
        for i in range(execution_steps):
            optimizer.zero_grad()
            with nvtx.range("forward pass"):
                output = model(execution_inputs[i])  
            torch.cuda.synchronize()
            
            loss = cross_entropy(output, execution_targets[i])
            with nvtx.range("backward pass"):
                loss.backward()
            torch.cuda.synchronize()

            with nvtx.range("optimizer step"):
                optimizer.step()
            torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--execution-steps", type=int, default=10)
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
    
    profile(
        model=model, 
        optimizer=optimizer,
        device=device,
        warmup_steps=args.warmup_steps,
        execution_steps=args.execution_steps,
    )